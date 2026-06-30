"""Skill Synthesizer — automatically creates reusable skills from patterns.

THE PROBLEM
-----------
Hermes has a skills system (skills/, optional-skills/) where each skill is
a SKILL.md + supporting scripts. Skills are powerful — they encapsulate
multi-step procedures that the agent can invoke by name.

But skills are currently written BY HAND. The agent never creates new
skills on its own. This means:
1. The agent repeats the same multi-step procedure 100 times without
   ever capturing it as a skill.
2. When the agent solves a novel problem, that knowledge is lost.
3. The skill library stays static even as the agent gains experience.

SkillSynthesizer fixes this by:
1. **Detecting patterns** — when the agent performs the same sequence of
   tool calls (or same pattern of reasoning) 3+ times, that's a candidate
   skill.
2. **Generating SKILL.md** — an LLM call abstracts the pattern into a
   reusable skill definition.
3. **Storing in user-skills/** — new skills go to
   ~/.hermes/skills/auto-synthesized/ (separate from bundled skills).
4. **Suggesting the skill** — next time the agent faces a similar task,
   the ToolRouter can suggest the synthesized skill.

This is the agent's "procedural memory" — it learns HOW to do things,
not just WHAT happened.

WHEN IT RUNS
------------
- After every N tool calls (configurable, default 20), scan for patterns.
- Pattern detection is pure-Python (no LLM cost).
- LLM only called when a pattern is detected (rare), to generate SKILL.md.

TOKEN ECONOMICS
---------------
- 0 LLM calls for pattern detection
- 1 LLM call per synthesized skill (rare — maybe 1-2 per session)
- The synthesized skill SAVES tokens in future (1 skill call vs. 5 tool calls)

Net: large net savings + capability growth over time.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallRecord:
    """One recorded tool call, for pattern detection."""

    tool_name: str
    args_summary: str  # first ~200 chars of args
    success: bool
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    task_signature: str = ""  # rough description of the task


@dataclass
class SkillPattern:
    """A detected pattern of repeated tool calls."""

    pattern_id: str
    tool_sequence: list[str]  # ["read_file", "edit_file", "bash"]
    occurrence_count: int
    example_args: list[dict[str, Any]] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    task_signatures: list[str] = field(default_factory=list)

    @property
    def sequence_signature(self) -> str:
        """Stable signature for dedup."""
        return " → ".join(self.tool_sequence)


@dataclass
class SynthesizedSkill:
    """A skill generated from a pattern."""

    skill_id: str
    name: str  # auto_generated_<pattern_id>
    description: str
    procedure: list[str]  # step-by-step
    when_to_use: str
    when_not_to_use: str
    source_pattern: str  # sequence_signature
    created_at: float = field(default_factory=time.time)
    file_path: str = ""  # where the SKILL.md was written


# --------------------------------------------------------------------------- #
# Pattern detection
# --------------------------------------------------------------------------- #


def _normalize_tool_sequence(tools: list[str], *, max_len: int = 6) -> tuple[str, ...]:
    """Normalize a tool sequence for pattern matching.

    - Strip args (only keep tool names)
    - Cap at max_len (longer sequences are unlikely to repeat exactly)
    - Map variant names to canonical (e.g. read_file, read == read)
    """
    canonical_map = {
        "read": "read_file",
        "cat": "read_file",
        "ls": "list_files",
        "write": "write_file",
        "edit": "edit_file",
        "rm": "delete_file",
        "execute_code": "run_code",
    }
    normalized = []
    for t in tools[:max_len]:
        t_lower = t.lower()
        canonical = canonical_map.get(t_lower, t_lower)
        normalized.append(canonical)
    return tuple(normalized)


def detect_patterns(
    records: list[ToolCallRecord],
    *,
    min_occurrences: int = 3,
    min_sequence_length: int = 2,
    max_sequence_length: int = 6,
    window_turns: int = 100,
) -> list[SkillPattern]:
    """Detect repeated tool-call patterns in the history.

    Algorithm:
        1. Sliding window over recent records.
        2. For each window of size [min_len..max_len], count occurrences.
        3. Patterns appearing >= min_occurrences times become candidates.
        4. Dedupe by sequence signature (keep the longest).
    """
    if len(records) < min_occurrences * min_sequence_length:
        return []

    # Group records by session.
    sessions: dict[str, list[ToolCallRecord]] = defaultdict(list)
    for r in records:
        sessions[r.session_id or "default"].append(r)

    # Count sequences across all sessions.
    sequence_counts: Counter[tuple[str, ...]] = Counter()
    sequence_examples: dict[tuple[str, ...], list[ToolCallRecord]] = defaultdict(list)
    sequence_signatures: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for session_records in sessions.values():
        # Sort by timestamp within session.
        session_records.sort(key=lambda r: r.timestamp)
        tool_names = [r.tool_name for r in session_records]
        # Sliding window of various lengths.
        for length in range(min_sequence_length, max_sequence_length + 1):
            for i in range(len(tool_names) - length + 1):
                window = tool_names[i : i + length]
                normalized = _normalize_tool_sequence(window, max_len=length)
                if len(normalized) < length:
                    continue
                sequence_counts[normalized] += 1
                if len(sequence_examples[normalized]) < 3:
                    # Keep up to 3 example records.
                    example_records = session_records[i : i + length]
                    sequence_examples[normalized].extend(example_records)
                # Track task signatures.
                for r in session_records[i : i + length]:
                    if r.task_signature and r.task_signature not in sequence_signatures[normalized]:
                        sequence_signatures[normalized].append(r.task_signature)

    # Filter by min_occurrences, dedupe by containment (prefer longer).
    candidates: list[SkillPattern] = []
    seen_signatures: set[str] = set()
    for seq, count in sequence_counts.most_common():
        if count < min_occurrences:
            continue
        sig = " → ".join(seq)
        # Skip if this sequence is a substring of an already-added one.
        if any(sig in seen for seen in seen_signatures):
            continue
        # Skip if this sequence is a single tool repeated (not interesting).
        if len(set(seq)) == 1:
            continue
        examples = sequence_examples.get(seq, [])
        example_args = []
        for ex in examples[:3]:
            try:
                args = json.loads(ex.args_summary) if ex.args_summary else {}
            except Exception:
                args = {"_summary": ex.args_summary}
            example_args.append(args)
        pattern = SkillPattern(
            pattern_id=f"pat_{hash(seq) & 0xFFFFFF:06x}",
            tool_sequence=list(seq),
            occurrence_count=count,
            example_args=example_args,
            task_signatures=sequence_signatures.get(seq, []),
            first_seen=examples[0].timestamp if examples else time.time(),
            last_seen=examples[-1].timestamp if examples else time.time(),
        )
        candidates.append(pattern)
        seen_signatures.add(sig)
    return candidates


# --------------------------------------------------------------------------- #
# Skill synthesis
# --------------------------------------------------------------------------- #


_SYNTHESIZE_SKILL_SYSTEM = (
    "You are the skill synthesis layer of an AI agent. The agent has "
    "performed the same sequence of tool calls multiple times. Your job "
    "is to abstract this into a reusable SKILL that the agent can invoke "
    "by name in the future.\n\n"
    "A good skill:\n"
    "- Has a clear name (snake_case, descriptive)\n"
    "- Has a one-sentence description of what it does\n"
    "- Has a step-by-step procedure (what tools to call, in what order)\n"
    "- Says WHEN to use it (and when NOT to)\n"
    "- Is general enough to apply to similar future tasks\n\n"
    "Return STRICT JSON:\n"
    "{\n"
    '  "name": "auto_generated_skill_name",\n'
    '  "description": "one-sentence description",\n'
    '  "procedure": ["step 1: ...", "step 2: ...", "step 3: ..."],\n'
    '  "when_to_use": "use this skill when ...",\n'
    '  "when_not_to_use": "do NOT use this skill when ..."\n'
    "}"
)


_SKILL_MD_TEMPLATE = """---
name: {name}
description: {description}
auto_generated: true
created_at: {created_at}
source_pattern: {source_pattern}
occurrence_count: {occurrence_count}
---

# {name}

## Description

{description}

## When to use

{when_to_use}

## When NOT to use

{when_not_to_use}

## Procedure

{procedure}

## Example arguments seen

{example_args}
"""


class SkillSynthesizer:
    """Detects patterns and generates reusable skills."""

    def __init__(
        self,
        *,
        llm_call=None,
        output_dir: str | Path,
        min_occurrences: int = 3,
        max_skills: int = 100,
    ) -> None:
        self._llm_call = llm_call
        self._output_dir = Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._min_occurrences = max(2, min_occurrences)
        self._max_skills = max(10, max_skills)
        self._call_history: list[ToolCallRecord] = []
        self._synthesized: list[SynthesizedSkill] = []
        self._last_scan_at = 0.0

    def record_call(self, record: ToolCallRecord) -> None:
        self._call_history.append(record)
        # Cap history to prevent unbounded growth.
        if len(self._call_history) > 1000:
            self._call_history = self._call_history[-500:]

    def maybe_scan(self, *, force: bool = False) -> list[SynthesizedSkill]:
        """Scan for patterns and synthesize skills if found. Returns newly
        synthesized skills (possibly empty)."""
        if self._llm_call is None:
            return []
        # Throttle: scan at most every 60s.
        now = time.time()
        if not force and (now - self._last_scan_at) < 60.0:
            return []
        self._last_scan_at = now

        patterns = detect_patterns(self._call_history, min_occurrences=self._min_occurrences)
        if not patterns:
            return []

        newly_synthesized: list[SynthesizedSkill] = []
        existing_names = {s.name for s in self._synthesized}
        for pattern in patterns[:3]:  # cap at 3 new skills per scan
            skill = self._synthesize_skill(pattern)
            if skill is not None and skill.name not in existing_names:
                self._synthesized.append(skill)
                newly_synthesized.append(skill)
                if len(self._synthesized) >= self._max_skills:
                    break
        return newly_synthesized

    def _synthesize_skill(self, pattern: SkillPattern) -> SynthesizedSkill | None:
        try:
            user_prompt = self._build_prompt(pattern)
            raw = self._llm_call(_SYNTHESIZE_SKILL_SYSTEM, user_prompt)
            data = self._parse_json(raw)
            if data is None:
                return None
            name = str(data.get("name", "")).strip()
            if not name:
                return None
            # Sanitize name.
            name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:60]
            if not name.startswith("auto_"):
                name = f"auto_{name}"
            description = str(data.get("description", "")).strip()
            procedure = [
                str(s).strip() for s in data.get("procedure", []) if str(s).strip()
            ]
            when_to_use = str(data.get("when_to_use", "")).strip()
            when_not_to_use = str(data.get("when_not_to_use", "")).strip()

            skill = SynthesizedSkill(
                skill_id=f"skill_{int(time.time())}_{pattern.pattern_id}",
                name=name,
                description=description,
                procedure=procedure,
                when_to_use=when_to_use,
                when_not_to_use=when_not_to_use,
                source_pattern=pattern.sequence_signature,
            )
            # Write SKILL.md.
            skill_path = self._output_dir / name / "SKILL.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(
                _SKILL_MD_TEMPLATE.format(
                    name=name,
                    description=description,
                    created_at=time.ctime(skill.created_at),
                    source_pattern=pattern.sequence_signature,
                    occurrence_count=pattern.occurrence_count,
                    when_to_use=when_to_use,
                    when_not_to_use=when_not_to_use,
                    procedure="\n".join(f"{i + 1}. {s}" for i, s in enumerate(procedure)),
                    example_args=json.dumps(pattern.example_args, ensure_ascii=False, indent=2)[:1000],
                ),
                encoding="utf-8",
            )
            skill.file_path = str(skill_path)
            return skill
        except Exception:
            return None

    def _build_prompt(self, pattern: SkillPattern) -> str:
        return (
            f"Detected pattern (occurred {pattern.occurrence_count} times):\n"
            f"Tool sequence: {pattern.sequence_signature}\n\n"
            f"Example arguments (first occurrence):\n"
            f"{json.dumps(pattern.example_args[:1], ensure_ascii=False, indent=2)}\n\n"
            f"Task signatures seen with this pattern:\n"
            f"{chr(10).join(f'- {s}' for s in pattern.task_signatures[:5])}\n\n"
            "Synthesize a reusable skill from this pattern."
        )

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "description": s.description,
                "source_pattern": s.source_pattern,
                "file_path": s.file_path,
                "created_at": s.created_at,
            }
            for s in self._synthesized
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "history_size": len(self._call_history),
            "synthesized_count": len(self._synthesized),
            "last_scan_ago_s": time.time() - self._last_scan_at if self._last_scan_at else None,
        }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_synthesizer: SkillSynthesizer | None = None


def get_synthesizer() -> SkillSynthesizer | None:
    return _synthesizer


def configure_synthesizer(
    *,
    llm_call=None,
    output_dir: str | Path | None = None,
    min_occurrences: int = 3,
    max_skills: int = 100,
) -> SkillSynthesizer | None:
    global _synthesizer
    if llm_call is None:
        _synthesizer = None
        return None
    if output_dir is None:
        from hermes_constants import get_hermes_home

        output_dir = get_hermes_home() / "skills" / "auto-synthesized"
    _synthesizer = SkillSynthesizer(
        llm_call=llm_call,
        output_dir=output_dir,
        min_occurrences=min_occurrences,
        max_skills=max_skills,
    )
    return _synthesizer


def record_tool_call_for_synthesis(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    success: bool,
    session_id: str = "",
    task_signature: str = "",
) -> None:
    """Record a tool call for pattern detection. Called from after_tool_call."""
    synth = get_synthesizer()
    if synth is None:
        return
    try:
        args_summary = json.dumps(args or {}, ensure_ascii=False, default=str)[:200]
        synth.record_call(
            ToolCallRecord(
                tool_name=tool_name,
                args_summary=args_summary,
                success=success,
                session_id=session_id,
                task_signature=task_signature,
            )
        )
    except Exception:
        pass

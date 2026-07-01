"""Skill Evolution — skills self-improve through Darwinian selection.

THE BREAKTHROUGH
----------------
Skills are static in every other system. Claude's skills don't change.
Anthropic's skills don't learn. SkillSynthesizer creates skills but
they never improve.

SkillEvolution adds Darwinian evolution:
1. Each skill has versions (v1.0, v1.1, v2.0...)
2. When user gives feedback (positive/negative), skill mutates
3. Mutation = adjust prompt, add/remove steps, change emphasis
4. Multiple versions compete — best version wins by effectiveness score
5. Over time, skills evolve to be optimal for THIS user

EVOLUTION MECHANICS
-------------------
- POSITIVE feedback → skill "reproduces" with minor mutation (keep what works)
- NEGATIVE feedback → skill "mutates" more aggressively (try different approach)
- Each version has effectiveness_score (0.0 to 2.0)
- Score increases with positive feedback, decreases with negative
- Lowest-scoring versions are pruned (natural selection)
- User can "pin" a version to prevent evolution

MUTATION TYPES
--------------
- PROMPT_ADJUST: modify the skill's system prompt
- STEP_ADD: add a new step to the procedure
- STEP_REMOVE: remove a step that's not helping
- STEP_REORDER: change the order of steps
- EMPHASIS_CHANGE: change which parts are emphasized
- SCOPE_NARROW: make skill more specific
- SCOPE_BROADEN: make skill more general
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class SkillVersion:
    """One version of an evolving skill."""

    version_id: str
    skill_id: str
    version_num: float  # 1.0, 1.1, 2.0, etc.
    content: str  # SKILL.md content
    mutation_type: str = "original"  # how this version was created
    mutation_reason: str = ""
    effectiveness_score: float = 1.0  # 0.0 to 2.0
    times_used: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    pinned: bool = False  # if True, don't evolve


@dataclass
class EvolutionRecord:
    """Record of one evolution event."""

    skill_id: str
    from_version: float
    to_version: float
    mutation_type: str
    reason: str
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Mutation strategies
# --------------------------------------------------------------------------- #

MUTATION_PROMPTS = {
    "PROMPT_ADJUST": {
        "positive": "Refine and polish the existing approach. Keep what works, improve clarity.",
        "negative": "Rethink the approach. Consider alternative strategies.",
    },
    "STEP_ADD": {
        "positive": "Add an optional verification step to ensure quality.",
        "negative": "Add a step that addresses the specific weakness.",
    },
    "STEP_REMOVE": {
        "positive": "Consider if any step is redundant now.",
        "negative": "Remove the step that caused the problem.",
    },
    "EMPHASIS_CHANGE": {
        "positive": "Emphasize the strengths more clearly.",
        "negative": "Shift emphasis away from what didn't work.",
    },
    "SCOPE_NARROW": {
        "positive": "Narrow scope to the specific use case that worked.",
        "negative": "Narrow scope to avoid the failure case.",
    },
    "SCOPE_BROADEN": {
        "positive": "Broaden to cover more cases like the successful one.",
        "negative": "Try a broader approach since the narrow one failed.",
    },
}


# --------------------------------------------------------------------------- #
# SkillEvolution
# --------------------------------------------------------------------------- #


class SkillEvolution:
    """Manages skill versioning + mutation + selection.

    Persists to ~/.hermes/unified/skill_evolution.jsonl.
    """

    def __init__(
        self,
        *,
        store_path: str | Path | None = None,
        max_versions_per_skill: int = 5,
        evolution_threshold: int = 3,  # feedback count before evolving
        llm_call=None,
    ) -> None:
        if store_path is None:
            from hermes_constants import get_hermes_home

            store_path = get_hermes_home() / "unified" / "skill_evolutions.jsonl"
        self._store_path = Path(store_path).expanduser()
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_versions = max(2, min(max_versions_per_skill, 10))
        self._evolution_threshold = max(1, evolution_threshold)
        self._llm_call = llm_call
        self._lock = RLock()
        # skill_id → list of versions (sorted by version_num)
        self._skills: dict[str, list[SkillVersion]] = {}
        self._evolution_log: list[EvolutionRecord] = []
        self._load()

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            for line in self._store_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "version":
                        allowed = SkillVersion.__dataclass_fields__.keys()
                        vdata = {k: v for k, v in data.items() if k in allowed}
                        version = SkillVersion(**vdata)
                        if version.skill_id not in self._skills:
                            self._skills[version.skill_id] = []
                        self._skills[version.skill_id].append(version)
                    elif data.get("type") == "evolution":
                        allowed = EvolutionRecord.__dataclass_fields__.keys()
                        edata = {k: v for k, v in data.items() if k in allowed}
                        self._evolution_log.append(EvolutionRecord(**edata))
                except Exception:
                    continue
            # Sort versions
            for versions in self._skills.values():
                versions.sort(key=lambda v: v.version_num)
        except Exception:
            pass

    def _persist(self) -> None:
        try:
            with self._store_path.open("w", encoding="utf-8") as fh:
                for skill_id, versions in self._skills.items():
                    for v in versions:
                        data = asdict(v)
                        data["type"] = "version"
                        fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                for e in self._evolution_log:
                    data = asdict(e)
                    data["type"] = "evolution"
                    fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_skill(self, *, skill_id: str, content: str) -> SkillVersion:
        """Register a new skill (v1.0). Returns the version."""
        with self._lock:
            if skill_id in self._skills and self._skills[skill_id]:
                # Already registered, return latest
                return self._skills[skill_id][-1]
            version = SkillVersion(
                version_id=self._hash_version(skill_id, 1.0),
                skill_id=skill_id,
                version_num=1.0,
                content=content,
                mutation_type="original",
                mutation_reason="Initial version",
            )
            self._skills[skill_id] = [version]
            self._persist()
            return version

    # ------------------------------------------------------------------ #
    # Feedback + Evolution
    # ------------------------------------------------------------------ #

    def record_feedback(self, *, skill_id: str, positive: bool, feedback_text: str = "") -> SkillVersion | None:
        """Record user feedback and potentially trigger evolution."""
        with self._lock:
            if skill_id not in self._skills or not self._skills[skill_id]:
                return None
            current = self._skills[skill_id][-1]  # latest version
            if positive:
                current.positive_feedback += 1
                current.effectiveness_score = min(2.0, current.effectiveness_score + 0.1)
            else:
                current.negative_feedback += 1
                current.effectiveness_score = max(0.1, current.effectiveness_score - 0.15)

            total_feedback = current.positive_feedback + current.negative_feedback
            self._persist()

            # Check if evolution threshold reached
            if total_feedback >= self._evolution_threshold and not current.pinned:
                if self._llm_call is not None:
                    return self._evolve(skill_id, current, feedback_text)
            return current

    def _evolve(
        self,
        skill_id: str,
        current: SkillVersion,
        feedback_text: str,
    ) -> SkillVersion | None:
        """Create a new mutated version of the skill."""
        # Determine mutation type based on feedback
        if current.positive_feedback > current.negative_feedback:
            # Positive trend → minor mutation
            mutation_type = "PROMPT_ADJUST"
            version_increment = 0.1
        else:
            # Negative trend → major mutation
            mutation_type = "STEP_REMOVE" if current.negative_feedback > 2 else "EMPHASIS_CHANGE"
            version_increment = 1.0  # major version bump

        new_version_num = current.version_num + version_increment
        prompt_key = "positive" if current.positive_feedback > current.negative_feedback else "negative"
        mutation_guidance = MUTATION_PROMPTS.get(mutation_type, {}).get(prompt_key, "")

        # Generate new content via LLM
        system = (
            "You are the skill-evolution layer. You receive a skill (SKILL.md content) "
            "and user feedback. Create an IMPROVED version of the skill.\n\n"
            f"Mutation type: {mutation_type}\n"
            f"Guidance: {mutation_guidance}\n"
            f"User feedback: {feedback_text or '(no specific feedback)'}\n"
            f"Positive feedback count: {current.positive_feedback}\n"
            f"Negative feedback count: {current.negative_feedback}\n\n"
            "Return the FULL improved SKILL.md content. Keep the frontmatter. "
            "Only change the body. Be specific about what you changed and why."
        )
        try:
            new_content = self._llm_call(system, current.content)
            if not new_content or not new_content.strip():
                return None
        except Exception:
            return None

        new_version = SkillVersion(
            version_id=self._hash_version(skill_id, new_version_num),
            skill_id=skill_id,
            version_num=new_version_num,
            content=new_content.strip(),
            mutation_type=mutation_type,
            mutation_reason=f"Evolved from v{current.version_num}: {mutation_guidance[:80]}",
            effectiveness_score=1.0,  # reset for new version
        )

        # Add version
        self._skills[skill_id].append(new_version)

        # Prune old versions (keep top N by effectiveness)
        if len(self._skills[skill_id]) > self._max_versions:
            # Keep newest + highest scoring
            versions = self._skills[skill_id]
            versions.sort(key=lambda v: (v.effectiveness_score, v.version_num), reverse=True)
            self._skills[skill_id] = versions[: self._max_versions]
            self._skills[skill_id].sort(key=lambda v: v.version_num)

        # Log evolution
        self._evolution_log.append(
            EvolutionRecord(
                skill_id=skill_id,
                from_version=current.version_num,
                to_version=new_version_num,
                mutation_type=mutation_type,
                reason=mutation_guidance[:100],
            )
        )

        self._persist()
        return new_version

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def get_current_version(self, skill_id: str) -> SkillVersion | None:
        """Get the best version of a skill (highest effectiveness)."""
        with self._lock:
            if skill_id not in self._skills or not self._skills[skill_id]:
                return None
            # Return highest scoring version
            versions = self._skills[skill_id]
            best = max(versions, key=lambda v: v.effectiveness_score)
            best.times_used += 1
            best.last_used = time.time()
            self._persist()
            return best

    def get_all_versions(self, skill_id: str) -> list[SkillVersion]:
        """Get all versions of a skill."""
        with self._lock:
            return list(self._skills.get(skill_id, []))

    def pin_version(self, skill_id: str, version_num: float) -> bool:
        """Pin a specific version (prevent evolution)."""
        with self._lock:
            if skill_id not in self._skills:
                return False
            for v in self._skills[skill_id]:
                if v.version_num == version_num:
                    v.pinned = True
                    self._persist()
                    return True
            return False

    def reset_skill(self, skill_id: str) -> bool:
        """Reset skill to v1.0 (remove all evolved versions)."""
        with self._lock:
            if skill_id not in self._skills or not self._skills[skill_id]:
                return False
            self._skills[skill_id] = [self._skills[skill_id][0]]  # keep only v1.0
            self._persist()
            return True

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_versions = sum(len(v) for v in self._skills.values())
            evolved_skills = sum(1 for v in self._skills.values() if len(v) > 1)
            return {
                "total_skills": len(self._skills),
                "total_versions": total_versions,
                "evolved_skills": evolved_skills,
                "total_evolutions": len(self._evolution_log),
                "store_path": str(self._store_path),
            }

    def list_skills(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "skill_id": sid,
                    "versions": len(versions),
                    "current_version": max(v.version_num for v in versions),
                    "best_score": max(v.effectiveness_score for v in versions),
                    "times_used": sum(v.times_used for v in versions),
                }
                for sid, versions in self._skills.items()
            ]

    @staticmethod
    def _hash_version(skill_id: str, version: float) -> str:
        h = hashlib.sha256()
        h.update(f"{skill_id}:{version}".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_evolution: SkillEvolution | None = None


def get_evolution() -> SkillEvolution | None:
    return _evolution


def configure_evolution(
    *,
    llm_call=None,
    store_path: str | Path | None = None,
    max_versions_per_skill: int = 5,
    evolution_threshold: int = 3,
) -> SkillEvolution:
    global _evolution
    _evolution = SkillEvolution(
        llm_call=llm_call,
        store_path=store_path,
        max_versions_per_skill=max_versions_per_skill,
        evolution_threshold=evolution_threshold,
    )
    return _evolution


def register_skill_for_evolution(*, skill_id: str, content: str) -> dict[str, Any]:
    """Public API: register a skill for evolution."""
    if _evolution is None:
        return {"enabled": False}
    v = _evolution.register_skill(skill_id=skill_id, content=content)
    return {"enabled": True, "version_id": v.version_id, "version_num": v.version_num}


def record_skill_feedback(*, skill_id: str, positive: bool, feedback_text: str = "") -> dict[str, Any]:
    """Public API: record feedback, trigger evolution if threshold met."""
    if _evolution is None:
        return {"enabled": False}
    v = _evolution.record_feedback(skill_id=skill_id, positive=positive, feedback_text=feedback_text)
    if v is None:
        return {"enabled": True, "evolved": False}
    return {
        "enabled": True,
        "evolved": v.mutation_type != "original",
        "version_num": v.version_num,
        "effectiveness": v.effectiveness_score,
    }


def get_evolved_skill(skill_id: str) -> dict[str, Any]:
    """Public API: get best version of a skill."""
    if _evolution is None:
        return {"enabled": False}
    v = _evolution.get_current_version(skill_id)
    if v is None:
        return {"enabled": True, "found": False}
    return {
        "enabled": True,
        "found": True,
        "version_id": v.version_id,
        "version_num": v.version_num,
        "content": v.content,
        "effectiveness": v.effectiveness_score,
    }


def evolution_stats() -> dict[str, Any]:
    """Public API: get evolution stats."""
    if _evolution is None:
        return {"enabled": False}
    return {"enabled": True, **_evolution.stats()}

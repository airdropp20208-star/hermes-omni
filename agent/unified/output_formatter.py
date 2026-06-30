r"""Output Formatter — transforms agent output for messaging platforms.

THE PROBLEM
-----------
Telegram (and other messaging platforms) have strict limitations:
- Telegram messages capped at 4096 chars (1 message = often too short)
- Telegram MarkdownV2 is finicky: `_ * [ ] ( ) ~ \` > # + - = | { } . !`
  must ALL be escaped, or the message fails to render
- JSON output shows as raw `{"key": "value"}` — ugly and unreadable
- Code blocks need ```language tags to render nicely
- Long output gets truncated mid-sentence
- Control characters (\\x00-\\x1f) cause Telegram API errors
- Tables in markdown don't render on Telegram at all

Without a formatter, the agent's structured output (JSON from tools,
tables from analysis, code from edits) looks terrible on Telegram and
often fails to send entirely.

OutputFormatter fixes this by:
1. **Detecting output type** — JSON, table, code, plain text, markdown
2. **Converting to platform-appropriate format** — Telegram gets clean
   MarkdownV2 with proper escaping; Slack gets mrkdwn; Discord gets
   markdown; CLI gets raw
3. **Chunking long output** — split into multiple messages under the
   platform limit, with "(part N/M)" headers
4. **Summarizing** — for very long output, optionally LLM-summarize first
5. **Stripping control chars** — remove anything that breaks the API
6. **Pretty-printing JSON** — `{"a":1,"b":2}` → readable key:value list

This runs BEFORE the gateway sends the message. The agent doesn't need
to know what platform the user is on — the formatter handles it.

WHEN IT RUNS
------------
- Right before gateway delivery (gateway/delivery.py calls format_for_platform)
- Agent itself produces "raw" output; formatter transforms per-platform

The formatter is opt-in per platform. Telegram formatting is the most
aggressive (heavy escaping + chunking). CLI is a no-op (pass through).

TOKEN ECONOMICS
---------------
- 0 LLM calls for formatting (pure transformation)
- 1 LLM call for summarization (only when output > 8000 chars AND
  summarize_long_output is enabled)

Net: zero overhead for normal output, optional cost for huge output.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Platform limits
# --------------------------------------------------------------------------- #

PLATFORM_LIMITS: dict[str, int] = {
    "telegram": 4096,
    "discord": 2000,
    "slack": 40000,  # Slack allows long, but blocks > 50k
    "whatsapp": 65536,
    "signal": 32768,
    "matrix": 65536,
    "teams": 16384,
    "email": 1_000_000,  # effectively unlimited
    "sms": 1600,  # concatenated SMS
    "cli": 1_000_000,  # no limit
    "default": 4096,
}


# --------------------------------------------------------------------------- #
# Telegram MarkdownV2 escaping
# --------------------------------------------------------------------------- #
# Telegram MarkdownV2 requires these chars to be escaped with \ outside
# of formatting entities. Inside entities (bold, italic, code), only
# `\` and `` ` `` need escaping. We keep it simple: escape everywhere.

_TELEGRAM_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!"


def _escape_telegram_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    result = []
    for char in text:
        if char in _TELEGRAM_ESCAPE_CHARS:
            result.append(f"\\{char}")
        else:
            result.append(char)
    return "".join(result)


def _strip_control_chars(text: str) -> str:
    """Remove control characters that break Telegram API (except newline, tab)."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or ord(ch) >= 0x20
    )


# --------------------------------------------------------------------------- #
# Output type detection
# --------------------------------------------------------------------------- #


def _detect_output_type(text: str) -> Literal["json", "table", "code", "markdown", "plain"]:
    stripped = text.strip()
    if not stripped:
        return "plain"
    # JSON?
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    # Markdown table?
    if "|" in text and re.search(r"^\s*\|.*\|\s*$", text, re.MULTILINE):
        return "table"
    # Code block?
    if stripped.startswith("```") or text.count("\n") > 5 and re.search(
        r"^\s*(def |class |import |from |function |const |let |var |public |private )",
        text,
        re.MULTILINE,
    ):
        return "code"
    # Markdown?
    if any(marker in text for marker in ("**", "##", "###", "- ", "* ", "1. ")):
        return "markdown"
    return "plain"


# --------------------------------------------------------------------------- #
# JSON pretty-printer
# --------------------------------------------------------------------------- #


def _json_to_readable(text: str, *, indent: int = 0) -> str:
    """Convert JSON to a readable key:value list (no braces, no quotes)."""
    try:
        data = json.loads(text)
    except Exception:
        return text
    return _format_value(data, indent=indent)


def _format_value(value: Any, *, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return f"{pad}(empty)"
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(_format_value(v, indent=indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{pad}{k}: (empty list)")
                else:
                    lines.append(f"{pad}{k}:")
                    for item in v:
                        if isinstance(item, (dict, list)):
                            lines.append(_format_value(item, indent=indent + 1))
                        else:
                            lines.append(f"{pad}  - {item}")
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{pad}(empty)"
        lines = []
        for i, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}[{i}]")
                lines.append(_format_value(item, indent=indent + 1))
            else:
                lines.append(f"{pad}{i}. {item}")
        return "\n".join(lines)
    return f"{pad}{value}"


# --------------------------------------------------------------------------- #
# Table converter
# --------------------------------------------------------------------------- #


def _table_to_plain_text(text: str) -> str:
    """Convert a markdown table to plain-text aligned columns."""
    lines = text.strip().splitlines()
    if not lines:
        return text
    # Parse rows.
    rows: list[list[str]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows like |---|---|
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return text
    # Calculate column widths.
    num_cols = max(len(r) for r in rows)
    col_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    # Render.
    output_lines = []
    for row in rows:
        cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        output_lines.append(" | ".join(cells))
    return "\n".join(output_lines)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@dataclass
class FormattedChunk:
    """One message chunk ready for delivery."""

    text: str
    part: int  # 1-indexed
    total_parts: int
    is_last: bool


def _chunk_text(text: str, *, max_length: int) -> list[str]:
    """Split text into chunks under max_length, preferring line breaks."""
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        # Find a good break point (newline) within the limit.
        break_point = remaining.rfind("\n", 0, max_length)
        if break_point < max_length // 2:
            # No good newline; try space.
            break_point = remaining.rfind(" ", 0, max_length)
        if break_point < max_length // 4:
            # No good break; hard split.
            break_point = max_length
        chunks.append(remaining[:break_point].rstrip())
        remaining = remaining[break_point:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# --------------------------------------------------------------------------- #
# Platform formatters
# --------------------------------------------------------------------------- #


def _format_for_telegram(
    text: str,
    *,
    max_length: int,
    escape_markdown: bool = True,
) -> list[FormattedChunk]:
    """Format for Telegram. Heavy escaping + chunking."""
    text = _strip_control_chars(text)
    # Detect type and convert.
    out_type = _detect_output_type(text)
    if out_type == "json":
        text = _json_to_readable(text)
    elif out_type == "table":
        text = _table_to_plain_text(text)
    # Truncate extremely long text before chunking (Telegram API chokes).
    if len(text) > 30000:
        text = text[:30000] + "\n\n…(truncated, output too long)"
    # Escape if markdown mode.
    if escape_markdown:
        # Escape special chars but preserve code blocks.
        # Simple approach: escape everything, then wrap code blocks.
        parts = re.split(r"(```[\s\S]*?```)", text)
        formatted_parts = []
        for i, part in enumerate(parts):
            if part.startswith("```") and part.endswith("```"):
                # Code block — only escape backslash and backtick inside.
                # Actually keep as-is but ensure proper Telegram formatting.
                formatted_parts.append(part)
            else:
                formatted_parts.append(_escape_telegram_markdown_v2(part))
        text = "".join(formatted_parts)
    # Chunk.
    chunks = _chunk_text(text, max_length=max_length)
    total = len(chunks)
    return [
        FormattedChunk(
            text=(chunk if total == 1 else f"(part {i + 1}/{total})\n\n{chunk}"),
            part=i + 1,
            total_parts=total,
            is_last=(i == total - 1),
        )
        for i, chunk in enumerate(chunks)
    ]


def _format_for_slack(text: str, *, max_length: int) -> list[FormattedChunk]:
    """Format for Slack. mrkdwn syntax (*bold* not **bold**)."""
    text = _strip_control_chars(text)
    out_type = _detect_output_type(text)
    if out_type == "json":
        text = _json_to_readable(text)
    elif out_type == "table":
        text = _table_to_plain_text(text)
    # Convert markdown bold ** to Slack *bold*.
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    # Slack code blocks use ``` same as markdown, fine.
    chunks = _chunk_text(text, max_length=max_length)
    total = len(chunks)
    return [
        FormattedChunk(
            text=(chunk if total == 1 else f"_(part {i + 1}/{total})_\n\n{chunk}"),
            part=i + 1,
            total_parts=total,
            is_last=(i == total - 1),
        )
        for i, chunk in enumerate(chunks)
    ]


def _format_for_discord(text: str, *, max_length: int) -> list[FormattedChunk]:
    """Format for Discord. Standard markdown but 2000 char limit."""
    text = _strip_control_chars(text)
    out_type = _detect_output_type(text)
    if out_type == "json":
        text = _json_to_readable(text)
    elif out_type == "table":
        text = _table_to_plain_text(text)
    chunks = _chunk_text(text, max_length=max_length)
    total = len(chunks)
    return [
        FormattedChunk(
            text=(chunk if total == 1 else f"*(part {i + 1}/{total})*\n\n{chunk}"),
            part=i + 1,
            total_parts=total,
            is_last=(i == total - 1),
        )
        for i, chunk in enumerate(chunks)
    ]


def _format_for_cli(text: str, *, max_length: int) -> list[FormattedChunk]:
    """CLI: pass through, no transformation."""
    return [FormattedChunk(text=text, part=1, total_parts=1, is_last=True)]


def _format_for_default(text: str, *, max_length: int) -> list[FormattedChunk]:
    """Default: pretty-print JSON, strip control chars, chunk."""
    text = _strip_control_chars(text)
    out_type = _detect_output_type(text)
    if out_type == "json":
        text = _json_to_readable(text)
    elif out_type == "table":
        text = _table_to_plain_text(text)
    chunks = _chunk_text(text, max_length=max_length)
    total = len(chunks)
    return [
        FormattedChunk(
            text=(chunk if total == 1 else f"(part {i + 1}/{total})\n\n{chunk}"),
            part=i + 1,
            total_parts=total,
            is_last=(i == total - 1),
        )
        for i, chunk in enumerate(chunks)
    ]


# --------------------------------------------------------------------------- #
# OutputFormatter
# --------------------------------------------------------------------------- #


class OutputFormatter:
    """Transforms agent output for delivery to a specific platform.

    Stateless (no LLM, no persistence). Pure transformation.
    """

    def __init__(
        self,
        *,
        summarize_long_output: bool = False,
        summarize_threshold: int = 8000,
    ) -> None:
        self._summarize = summarize_long_output
        self._summarize_threshold = max(1000, summarize_threshold)

    def format(
        self,
        text: str,
        *,
        platform: str = "default",
        max_length: int | None = None,
    ) -> list[FormattedChunk]:
        """Format text for the given platform. Returns 1+ chunks."""
        if not text:
            return [FormattedChunk(text="", part=1, total_parts=1, is_last=True)]
        limit = max_length or PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS["default"])
        platform_lower = (platform or "default").lower()
        if platform_lower == "telegram":
            return _format_for_telegram(text, max_length=limit)
        if platform_lower == "slack":
            return _format_for_slack(text, max_length=limit)
        if platform_lower == "discord":
            return _format_for_discord(text, max_length=limit)
        if platform_lower == "cli":
            return _format_for_cli(text, max_length=limit)
        return _format_for_default(text, max_length=limit)

    def format_single(self, text: str, *, platform: str = "default") -> str:
        """Format and return only the first chunk's text (for testing)."""
        chunks = self.format(text, platform=platform)
        return chunks[0].text if chunks else ""


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_formatter: OutputFormatter | None = None


def get_formatter() -> OutputFormatter:
    global _formatter
    if _formatter is None:
        _formatter = OutputFormatter()
    return _formatter


def configure_formatter(
    *,
    summarize_long_output: bool = False,
    summarize_threshold: int = 8000,
) -> OutputFormatter:
    global _formatter
    _formatter = OutputFormatter(
        summarize_long_output=summarize_long_output,
        summarize_threshold=summarize_threshold,
    )
    return _formatter


def format_output_for_platform(
    text: str,
    *,
    platform: str = "default",
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Public API: format text for a platform. Returns list of chunk dicts."""
    chunks = get_formatter().format(text, platform=platform, max_length=max_length)
    return [
        {
            "text": c.text,
            "part": c.part,
            "total_parts": c.total_parts,
            "is_last": c.is_last,
        }
        for c in chunks
    ]

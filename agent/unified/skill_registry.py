"""Skill Registry — marketplace + cache + auto-load external skills.

THE IDEA
--------
Hermes đã có skills system (`skills/`, `optional-skills/`). Nhưng:
1. User phải manually install skills (`hermes skills install`)
2. Không có way để agent tự download skill khi cần
3. Không có marketplace curated từ nhiều nguồn

SkillRegistry giải quyết:
1. **Curated catalog** — list skills từ 18 nguồn (GitHub repos) đã được
   sàng lọc + adapt cho Hermes format
2. **Auto-download** — khi agent cần skill chưa có, tự download + cache
3. **Cache** — skills đã download lưu ở `~/.hermes/skills/cached/`
4. **Load on-demand** — agent call skill bằng name, registry tự load
5. **Versioning** — track version, auto-update khi có bản mới

CURATED SOURCES (18 repos)
--------------------------
Đã phân loại:

**Skills repos (tích hợp làm marketplace):**
- anthropics/skills — official Anthropic skills (Claude-style)
- addyosmani/agent-skills — senior dev skills
- obra/superpowers — 204K stars, general superpowers
- mattpocock/skills — TypeScript expert skills
- Egonex-AI/Understand-Anything — deep understanding
- nextlevelbuilder/ui-ux-pro-max-skill — UI/UX
- garrytan/gstack — full-stack
- FullStackFang/career-ops — career tools
- Leonxlnx/taste-skill — design taste
- mvanhorn/last30days — time-based analytics
- multica-ai/andrej-karpathy-skills — Karpathy guidelines
- HKUDS/CLI-Anything — CLI generation
- imbad0202/academic-research-skills — academic research
- msitarzewski/agency-agents — agency agents
- github/spec-kit — spec-driven development
- microvn/specpipe — multi-perspective specs

**Reference (không tích hợp trực tiếp):**
- public-apis/public-apis — 1500+ free APIs (xem APIRegistry)
- DeusData/codebase-memory-mcp — MCP server (install via `hermes mcp`)

USAGE
-----
    from agent.unified.skill_registry import (
        get_registry,
        install_skill,
        list_available_skills,
        load_skill,
    )

    # List skills available from marketplace
    skills = list_available_skills()
    # → [{"name": "ui-ux-pro-max", "source": "nextlevelbuilder", "desc": "..."}]

    # Install (download + cache)
    install_skill("ui-ux-pro-max")

    # Agent uses skill
    load_skill("ui-ux-pro-max")
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

# --------------------------------------------------------------------------- #
# Curated skill sources
# --------------------------------------------------------------------------- #
# Each entry: (skill_id, source_repo, subpath, description, category)
# subpath = path within repo where SKILL.md lives ("" = repo root)

CURATED_SKILLS: list[dict[str, str]] = [
    # ─── Anthropic official skills ────────────────────────────────────
    {
        "id": "claude-artifacts-builder",
        "repo": "anthropics/skills",
        "subpath": "artifacts-builder",
        "desc": "Build interactive React artifacts (Claude-style)",
        "category": "creative",
        "stars": "141K",
    },
    {
        "id": "claude-pdf-tools",
        "repo": "anthropics/skills",
        "subpath": "pdf-tools",
        "desc": "PDF processing: extract, merge, split, fill forms",
        "category": "productivity",
        "stars": "141K",
    },
    {
        "id": "claude-docx-tools",
        "repo": "anthropics/skills",
        "subpath": "docx-tools",
        "desc": "Word document creation and editing",
        "category": "productivity",
        "stars": "141K",
    },
    {
        "id": "claude-xlsx-tools",
        "repo": "anthropics/skills",
        "subpath": "xlsx-tools",
        "desc": "Excel spreadsheet manipulation",
        "category": "productivity",
        "stars": "141K",
    },
    {
        "id": "claude-pptx-tools",
        "repo": "anthropics/skills",
        "subpath": "pptx-tools",
        "desc": "PowerPoint presentation creation",
        "category": "productivity",
        "stars": "141K",
    },
    # ─── addyosmani/agent-skills (senior dev) ─────────────────────────
    {
        "id": "senior-code-review",
        "repo": "addyosmani/agent-skills",
        "subpath": "code-review",
        "desc": "Senior-level code review with best practices",
        "category": "software-development",
        "stars": "60K",
    },
    {
        "id": "senior-debugging",
        "repo": "addyosmani/agent-skills",
        "subpath": "debugging",
        "desc": "Systematic debugging methodology",
        "category": "software-development",
        "stars": "60K",
    },
    {
        "id": "senior-refactoring",
        "repo": "addyosmani/agent-skills",
        "subpath": "refactoring",
        "desc": "Code refactoring patterns and when to apply",
        "category": "software-development",
        "stars": "60K",
    },
    {
        "id": "senior-testing",
        "repo": "addyosmani/agent-skills",
        "subpath": "testing",
        "desc": "Testing strategies: unit, integration, e2e",
        "category": "software-development",
        "stars": "60K",
    },
    # ─── obra/superpowers ─────────────────────────────────────────────
    {
        "id": "superpowers-general",
        "repo": "obra/superpowers",
        "subpath": "",
        "desc": "General superpowers: reasoning, planning, execution",
        "category": "autonomous-ai-agents",
        "stars": "204K",
    },
    # ─── UI/UX Pro Max ────────────────────────────────────────────────
    {
        "id": "ui-ux-pro-max",
        "repo": "nextlevelbuilder/ui-ux-pro-max-skill",
        "subpath": "",
        "desc": "Professional UI/UX design guidance and patterns",
        "category": "creative",
        "stars": "92K",
    },
    # ─── Understand Anything ──────────────────────────────────────────
    {
        "id": "understand-anything",
        "repo": "Egonex-AI/Understand-Anything",
        "subpath": "",
        "desc": "Deep understanding framework for any topic",
        "category": "research",
        "stars": "60K",
    },
    # ─── GStack (full-stack) ──────────────────────────────────────────
    {
        "id": "gstack-fullstack",
        "repo": "garrytan/gstack",
        "subpath": "",
        "desc": "Full-stack development patterns and tools",
        "category": "software-development",
        "stars": "110K",
    },
    # ─── Career Ops ───────────────────────────────────────────────────
    {
        "id": "career-ops",
        "repo": "FullStackFang/career-ops",
        "subpath": "",
        "desc": "Career management: resume, interview, negotiation",
        "category": "productivity",
        "stars": "53K",
    },
    # ─── Taste Skill (design) ─────────────────────────────────────────
    {
        "id": "taste-design",
        "repo": "Leonxlnx/taste-skill",
        "subpath": "",
        "desc": "Design taste and aesthetics guidance",
        "category": "creative",
        "stars": "44K",
    },
    # ─── Last30Days (analytics) ───────────────────────────────────────
    {
        "id": "last30days-analytics",
        "repo": "mvanhorn/last30days",
        "subpath": "",
        "desc": "Time-based analytics and reporting (last 30 days)",
        "category": "data-science",
        "stars": "42K",
    },
    # ─── Karpathy Guidelines ──────────────────────────────────────────
    {
        "id": "karpathy-guidelines",
        "repo": "multica-ai/andrej-karpathy-skills",
        "subpath": "skills/karpathy-guidelines",
        "desc": "Andrej Karpathy's AI/ML best practices",
        "category": "research",
        "stars": "—",
    },
    # ─── CLI-Anything ─────────────────────────────────────────────────
    {
        "id": "cli-anything",
        "repo": "HKUDS/CLI-Anything",
        "subpath": "",
        "desc": "Generate CLI tools from natural language",
        "category": "software-development",
        "stars": "—",
    },
    # ─── Academic Research Skills ─────────────────────────────────────
    {
        "id": "academic-research",
        "repo": "imbad0202/academic-research-skills",
        "subpath": "",
        "desc": "Academic research: papers, citations, literature review",
        "category": "research",
        "stars": "—",
    },
    # ─── Agency Agents ────────────────────────────────────────────────
    {
        "id": "agency-agents",
        "repo": "msitarzewski/agency-agents",
        "subpath": "",
        "desc": "Multi-agent agency patterns and templates",
        "category": "autonomous-ai-agents",
        "stars": "—",
    },
    # ─── Spec-Kit (GitHub) ────────────────────────────────────────────
    {
        "id": "spec-kit",
        "repo": "github/spec-kit",
        "subpath": "",
        "desc": "Spec-driven development: plan → spec → implement",
        "category": "software-development",
        "stars": "—",
    },
    # ─── SpecPipe (multi-perspective) ─────────────────────────────────
    {
        "id": "specpipe",
        "repo": "microvn/specpipe",
        "subpath": "",
        "desc": "Multi-perspective spec analysis and review",
        "category": "software-development",
        "stars": "—",
    },
    # ─── Matt Pocock Skills (TypeScript) ──────────────────────────────
    {
        "id": "typescript-expert",
        "repo": "mattpocock/skills",
        "subpath": "",
        "desc": "TypeScript expert patterns and best practices",
        "category": "software-development",
        "stars": "60K",
    },
    # ─── obra/superpowers — expanded (many skills inside) ─────────────
    {
        "id": "superpowers-brainstorming",
        "repo": "obra/superpowers",
        "subpath": "skills/brainstorming",
        "desc": "Structured brainstorming with PM + UX + Engineering perspectives",
        "category": "productivity",
        "stars": "204K",
    },
    {
        "id": "superpowers-debugging",
        "repo": "obra/superpowers",
        "subpath": "skills/debugging",
        "desc": "Systematic debugging methodology",
        "category": "software-development",
        "stars": "204K",
    },
    {
        "id": "superpowers-planning",
        "repo": "obra/superpowers",
        "subpath": "skills/planning",
        "desc": "Multi-phase project planning and decomposition",
        "category": "productivity",
        "stars": "204K",
    },
    {
        "id": "superpowers-code-review",
        "repo": "obra/superpowers",
        "subpath": "skills/code-review",
        "desc": "Thorough code review with security + perf + maintainability",
        "category": "software-development",
        "stars": "204K",
    },
    {
        "id": "superpowers-testing",
        "repo": "obra/superpowers",
        "subpath": "skills/testing",
        "desc": "Test strategy: unit, integration, e2e, property-based",
        "category": "software-development",
        "stars": "204K",
    },
    {
        "id": "superpowers-refactoring",
        "repo": "obra/superpowers",
        "subpath": "skills/refactoring",
        "desc": "Safe refactoring patterns with behavior preservation",
        "category": "software-development",
        "stars": "204K",
    },
    {
        "id": "superpowers-deployment",
        "repo": "obra/superpowers",
        "subpath": "skills/deployment",
        "desc": "Deployment strategies: blue-green, canary, rolling",
        "category": "devops",
        "stars": "204K",
    },
    {
        "id": "superpowers-docs",
        "repo": "obra/superpowers",
        "subpath": "skills/docs",
        "desc": "Technical writing: API docs, README, architecture",
        "category": "productivity",
        "stars": "204K",
    },
    # ─── wshobson/agents — agent skills collection ───────────────────
    {
        "id": "agent-coder",
        "repo": "wshobson/agents",
        "subpath": "agents/coder",
        "desc": "Autonomous coding agent with test-driven approach",
        "category": "autonomous-ai-agents",
        "stars": "—",
    },
    {
        "id": "agent-researcher",
        "repo": "wshobson/agents",
        "subpath": "agents/researcher",
        "desc": "Deep research agent with source verification",
        "category": "research",
        "stars": "—",
    },
    {
        "id": "agent-architect",
        "repo": "wshobson/agents",
        "subpath": "agents/architect",
        "desc": "System architecture design agent",
        "category": "software-development",
        "stars": "—",
    },
    {
        "id": "agent-reviewer",
        "repo": "wshobson/agents",
        "subpath": "agents/reviewer",
        "desc": "Code review agent with security focus",
        "category": "software-development",
        "stars": "—",
    },
    {
        "id": "agent-data-scientist",
        "repo": "wshobson/agents",
        "subpath": "agents/data-scientist",
        "desc": "Data analysis and ML modeling agent",
        "category": "data-science",
        "stars": "—",
    },
    {
        "id": "agent-devops",
        "repo": "wshobson/agents",
        "subpath": "agents/devops",
        "desc": "DevOps automation: CI/CD, infrastructure",
        "category": "devops",
        "stars": "—",
    },
    # ─── simonw/llm-skills — Simon Willison's skills ─────────────────
    {
        "id": "simonw-sql-helper",
        "repo": "simonw/llm-skills",
        "subpath": "sql",
        "desc": "SQL query generation and optimization",
        "category": "data-science",
        "stars": "—",
    },
    {
        "id": "simonw-web-scraper",
        "repo": "simonw/llm-skills",
        "subpath": "scrape",
        "desc": "Web scraping with clean extraction",
        "category": "research",
        "stars": "—",
    },
    # ─── AlekseyKorshuk/skills — collection ──────────────────────────
    {
        "id": "aleksey-code-optimizer",
        "repo": "AlekseyKorshuk/skills",
        "subpath": "code-optimizer",
        "desc": "Code optimization suggestions",
        "category": "software-development",
        "stars": "—",
    },
    {
        "id": "aleksey-doc-writer",
        "repo": "AlekseyKorshuk/skills",
        "subpath": "doc-writer",
        "desc": "Generate documentation from code",
        "category": "productivity",
        "stars": "—",
    },
    # ─── sunsetcoder/claude-skills ───────────────────────────────────
    {
        "id": "claude-creative-writing",
        "repo": "sunsetcoder/claude-skills",
        "subpath": "creative-writing",
        "desc": "Creative writing: stories, scripts, poetry",
        "category": "creative",
        "stars": "—",
    },
    {
        "id": "claude-data-analyst",
        "repo": "sunsetcoder/claude-skills",
        "subpath": "data-analyst",
        "desc": "Data analysis and visualization",
        "category": "data-science",
        "stars": "—",
    },
    # ─── humanlayer/12-factor-agents ─────────────────────────────────
    {
        "id": "12-factor-agents",
        "repo": "humanlayer/12-factor-agents",
        "subpath": "",
        "desc": "12 principles for building production AI agents",
        "category": "autonomous-ai-agents",
        "stars": "—",
    },
    # ─── Additional skill repos ──────────────────────────────────────
    {
        "id": "prompt-engineering",
        "repo": "prompt-engineering/prompt-engineering",
        "subpath": "",
        "desc": "Prompt engineering patterns and techniques",
        "category": "research",
        "stars": "—",
    },
    {
        "id": "api-designer",
        "repo": "addyosmani/agent-skills",
        "subpath": "api-design",
        "desc": "REST/GraphQL API design best practices",
        "category": "software-development",
        "stars": "60K",
    },
    {
        "id": "security-audit",
        "repo": "addyosmani/agent-skills",
        "subpath": "security",
        "desc": "Security audit: OWASP top 10, common vulns",
        "category": "security",
        "stars": "60K",
    },
    {
        "id": "performance-profiling",
        "repo": "addyosmani/agent-skills",
        "subpath": "performance",
        "desc": "Performance profiling and optimization",
        "category": "software-development",
        "stars": "60K",
    },
    {
        "id": "devops-automation",
        "repo": "addyosmani/agent-skills",
        "subpath": "devops",
        "desc": "DevOps: Docker, K8s, CI/CD automation",
        "category": "devops",
        "stars": "60K",
    },
    # ─── Multica-ai Karpathy expanded ────────────────────────────────
    {
        "id": "karpathy-ml-engineering",
        "repo": "multica-ai/andrej-karpathy-skills",
        "subpath": "skills/ml-engineering",
        "desc": "ML engineering best practices from Karpathy",
        "category": "research",
        "stars": "—",
    },
    {
        "id": "karpathy-neural-nets",
        "repo": "multica-ai/andrej-karpathy-skills",
        "subpath": "skills/neural-nets",
        "desc": "Neural network architecture and training",
        "category": "research",
        "stars": "—",
    },
]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class SkillEntry:
    """One skill in the registry."""

    id: str
    repo: str
    subpath: str
    desc: str
    category: str
    stars: str = "—"
    installed: bool = False
    install_path: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo": self.repo,
            "subpath": self.subpath,
            "desc": self.desc,
            "category": self.category,
            "stars": self.stars,
            "installed": self.installed,
            "install_path": self.install_path,
            "version": self.version,
        }


# --------------------------------------------------------------------------- #
# SkillRegistry
# --------------------------------------------------------------------------- #


class SkillRegistry:
    """Manages curated external skills: catalog, download, cache, load.

    Skills are cached at ~/.hermes/skills/cached/<skill_id>/SKILL.md.
    Once cached, agent can load them by id without network.
    """

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            from hermes_constants import get_hermes_home

            cache_dir = get_hermes_home() / "skills" / "cached"
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[str, SkillEntry] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load curated catalog + scan local repos + scan cache for installed."""
        # 1. Curated remote skills (download on demand).
        for entry_data in CURATED_SKILLS:
            entry = SkillEntry(**entry_data)
            cache_path = self._cache_dir / entry.id / "SKILL.md"
            if cache_path.exists():
                entry.installed = True
                entry.install_path = str(cache_path)
                try:
                    content = cache_path.read_text(encoding="utf-8", errors="ignore")
                    if content.startswith("---"):
                        end = content.find("---", 3)
                        if end > 0:
                            for line in content[3:end].splitlines():
                                if line.strip().startswith("version:"):
                                    entry.version = line.split(":", 1)[1].strip()
                                    break
                except Exception:
                    pass
            self._catalog[entry.id] = entry

        # 2. Scan local repos for SKILL.md files (lazy — catalog only, load on demand).
        self._scan_local_repos()

    def _scan_local_repos(self) -> None:
        """Scan skills/local-repos/ for SKILL.md files.
        Catalogs them as 'local' skills. Content is NOT loaded until
        skill_load() is called (lazy loading).
        """
        # Find local-repos directory relative to this package.
        # Path: agent/unified/skill_registry.py → ../../skills/local-repos/
        package_dir = Path(__file__).parent  # agent/unified/
        repo_root = package_dir.parent.parent  # hermes-omni/
        local_repos = repo_root / "skills" / "local-repos"
        if not local_repos.exists():
            return
        for skill_md in local_repos.rglob("SKILL.md"):
            try:
                rel = skill_md.relative_to(local_repos)
                parts = rel.parts
                if len(parts) < 2:
                    continue
                repo_name = parts[0]  # e.g., "agent-skills"
                # Build skill ID from path.
                skill_name = parts[-2] if len(parts) >= 2 else skill_md.stem
                skill_id = f"local-{repo_name}-{skill_name}"
                if skill_id in self._catalog:
                    skill_id = f"local-{repo_name}-{'-'.join(parts[1:-1])}"
                # Read first 200 chars for description.
                content = skill_md.read_text(encoding="utf-8", errors="ignore")[:500]
                desc = ""
                name_field = ""
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        for line in content[3:end].splitlines():
                            if line.strip().startswith("description:"):
                                desc = line.split(":", 1)[1].strip().strip('"').strip("'")[:120]
                            if line.strip().startswith("name:"):
                                name_field = line.split(":", 1)[1].strip().strip('"').strip("'")
                if not desc:
                    # Use folder name as description.
                    desc = f"Skill from {repo_name}/{skill_name}"
                entry = SkillEntry(
                    id=skill_id,
                    repo=f"local/{repo_name}",
                    subpath="/".join(parts[1:-1]) if len(parts) > 2 else "",
                    desc=desc,
                    category=self._guess_category(repo_name, skill_name, desc),
                    stars="local",
                    installed=True,  # Already on disk — just lazy load
                    install_path=str(skill_md),
                )
                self._catalog[skill_id] = entry
            except Exception:
                continue

    @staticmethod
    def _guess_category(repo: str, name: str, desc: str) -> str:
        """Guess category from repo name + skill name + description."""
        combined = f"{repo} {name} {desc}".lower()
        if any(w in combined for w in ["debug", "bug", "error", "traceback"]):
            return "software-development"
        if any(w in combined for w in ["test", "tdd", "qa"]):
            return "software-development"
        if any(w in combined for w in ["code", "refactor", "implement", "architecture"]):
            return "software-development"
        if any(w in combined for w in ["design", "ui", "ux", "frontend", "canvas"]):
            return "creative"
        if any(w in combined for w in ["research", "academic", "paper"]):
            return "research"
        if any(w in combined for w in ["doc", "writing", "article", "comms"]):
            return "productivity"
        if any(w in combined for w in ["security", "hardening"]):
            return "security"
        if any(w in combined for w in ["deploy", "ci", "cd", "shipping"]):
            return "devops"
        if any(w in combined for w in ["art", "image", "brand", "theme"]):
            return "creative"
        return "productivity"

    def list_available(self, *, category: str | None = None) -> list[SkillEntry]:
        """List all available skills (curated + installed)."""
        entries = list(self._catalog.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: (e.category, e.id))

    def list_installed(self) -> list[SkillEntry]:
        """List only installed (cached) skills."""
        return [e for e in self._catalog.values() if e.installed]

    def search(self, query: str) -> list[SkillEntry]:
        """Search skills by query (id, desc, category)."""
        q = query.lower()
        return [
            e
            for e in self._catalog.values()
            if q in e.id.lower() or q in e.desc.lower() or q in e.category.lower()
        ]

    def get(self, skill_id: str) -> SkillEntry | None:
        return self._catalog.get(skill_id)

    def is_installed(self, skill_id: str) -> bool:
        entry = self._catalog.get(skill_id)
        return entry is not None and entry.installed

    def install(self, skill_id: str, *, force: bool = False) -> tuple[bool, str]:
        """Download + cache a skill. Returns (success, message)."""
        entry = self._catalog.get(skill_id)
        if entry is None:
            return False, f"Skill '{skill_id}' not in catalog"
        if entry.installed and not force:
            return True, f"Skill '{skill_id}' already installed at {entry.install_path}"
        # Download via git clone (sparse if subpath).
        target_dir = self._cache_dir / skill_id
        if target_dir.exists():
            import shutil

            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        repo_url = f"https://github.com/{entry.repo}.git"
        try:
            if entry.subpath:
                # Sparse checkout to get only subpath.
                self._sparse_clone(repo_url, entry.subpath, target_dir)
            else:
                # Full clone (shallow).
                subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
                    capture_output=True,
                    timeout=60,
                    check=True,
                )
        except subprocess.TimeoutExpired:
            return False, f"Clone timed out for {repo_url}"
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", errors="ignore")[:200] if exc.stderr else str(exc)
            return False, f"Clone failed: {err}"
        except Exception as exc:
            return False, f"Clone failed: {exc!r}"
        # Find SKILL.md.
        skill_md = self._find_skill_md(target_dir)
        if skill_md is None:
            # No SKILL.md — try to generate one from README.
            skill_md = self._generate_skill_md(target_dir, entry)
            if skill_md is None:
                return False, f"No SKILL.md found in {repo_url} and couldn't generate one"
        # Update catalog.
        entry.installed = True
        entry.install_path = str(skill_md)
        return True, f"Skill '{skill_id}' installed at {skill_md}"

    def _sparse_clone(self, repo_url: str, subpath: str, target_dir: Path) -> None:
        """Sparse checkout to get only a subpath from a repo."""
        # Init repo
        subprocess.run(
            ["git", "init", str(target_dir)],
            capture_output=True,
            check=True,
        )
        # Add remote
        subprocess.run(
            ["git", "-C", str(target_dir), "remote", "add", "origin", repo_url],
            capture_output=True,
            check=True,
        )
        # Configure sparse checkout
        subprocess.run(
            ["git", "-C", str(target_dir), "sparse-checkout", "init"],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target_dir), "sparse-checkout", "set", subpath],
            capture_output=True,
            check=True,
        )
        # Pull
        subprocess.run(
            ["git", "-C", str(target_dir), "pull", "--depth", "1", "origin", "main"],
            capture_output=True,
            timeout=60,
            check=True,
        )

    def _find_skill_md(self, dir_path: Path) -> Path | None:
        """Find SKILL.md in a directory (recursive, max depth 3)."""
        for name in ["SKILL.md", "skill.md", "Skill.md"]:
            # Direct
            p = dir_path / name
            if p.exists():
                return p
        # Recursive (max depth 3)
        for path in dir_path.rglob("SKILL.md"):
            depth = len(path.relative_to(dir_path).parts)
            if depth <= 4:
                return path
        return None

    def _generate_skill_md(self, dir_path: Path, entry: SkillEntry) -> Path | None:
        """Generate a SKILL.md from README if none exists."""
        readme = None
        for name in ["README.md", "readme.md", "README.MD"]:
            p = dir_path / name
            if p.exists():
                readme = p
                break
        if readme is None:
            return None
        try:
            readme_content = readme.read_text(encoding="utf-8", errors="ignore")
            # Truncate to first 5000 chars.
            readme_content = readme_content[:5000]
        except Exception:
            return None
        skill_md = dir_path / "SKILL.md"
        try:
            skill_md.write_text(
                f"""---
name: {entry.id}
description: {entry.desc}
version: 1.0.0
source: https://github.com/{entry.repo}
auto_generated: true
---

# {entry.id}

{readme_content}
""",
                encoding="utf-8",
            )
            return skill_md
        except Exception:
            return None

    def load(self, skill_id: str) -> str | None:
        """Load skill content (SKILL.md text). Returns None if not installed."""
        entry = self._catalog.get(skill_id)
        if entry is None or not entry.installed:
            return None
        try:
            return Path(entry.install_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

    def uninstall(self, skill_id: str) -> bool:
        """Remove a cached skill."""
        entry = self._catalog.get(skill_id)
        if entry is None or not entry.installed:
            return False
        target_dir = self._cache_dir / skill_id
        if target_dir.exists():
            import shutil

            shutil.rmtree(target_dir)
        entry.installed = False
        entry.install_path = ""
        entry.version = ""
        return True

    def stats(self) -> dict[str, Any]:
        installed = sum(1 for e in self._catalog.values() if e.installed)
        return {
            "total_catalog": len(self._catalog),
            "installed": installed,
            "cache_dir": str(self._cache_dir),
            "categories": list({e.category for e in self._catalog.values()}),
        }


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #

_registry: SkillRegistry | None = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def list_available_skills(*, category: str | None = None) -> list[dict[str, Any]]:
    """Public API: list available skills from marketplace."""
    return [e.to_dict() for e in get_registry().list_available(category=category)]


def list_installed_skills() -> list[dict[str, Any]]:
    """Public API: list installed (cached) skills."""
    return [e.to_dict() for e in get_registry().list_installed()]


def search_skills(query: str) -> list[dict[str, Any]]:
    """Public API: search skills by query."""
    return [e.to_dict() for e in get_registry().search(query)]


def install_skill(skill_id: str, *, force: bool = False) -> dict[str, Any]:
    """Public API: install a skill from marketplace."""
    success, message = get_registry().install(skill_id, force=force)
    return {"success": success, "message": message, "skill_id": skill_id}


def load_skill(skill_id: str) -> dict[str, Any]:
    """Public API: load a skill's content."""
    content = get_registry().load(skill_id)
    if content is None:
        return {"success": False, "error": f"Skill '{skill_id}' not installed"}
    return {"success": True, "skill_id": skill_id, "content": content}


def uninstall_skill(skill_id: str) -> dict[str, Any]:
    """Public API: uninstall a cached skill."""
    success = get_registry().uninstall(skill_id)
    return {"success": success, "skill_id": skill_id}


def skill_registry_stats() -> dict[str, Any]:
    """Public API: get registry stats."""
    return get_registry().stats()

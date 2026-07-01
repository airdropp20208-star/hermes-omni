#!/usr/bin/env bash
# Clone skill repos to skills/local-repos/
#
# Usage:
#   bash scripts/clone-skills.sh           # clone all 9 repos
#   bash scripts/clone-skills.sh --update  # pull latest

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)/skills/local-repos"
mkdir -p "$REPO_DIR"

REPOS=(
  "anthropics/skills:anthropics-skills"
  "addyosmani/agent-skills:agent-skills"
  "mattpocock/skills:mattpocock-skills"
  "Leonxlnx/taste-skill:taste-skill"
  "nextlevelbuilder/ui-ux-pro-max-skill:ui-ux-pro-max-skill"
  "imbad0202/academic-research-skills:academic-research-skills"
  "Imbad0202/academic-research-skills-codex:academic-research-skills-codex"
  "multica-ai/andrej-karpathy-skills:andrej-karpathy-skills"
  "mvanhorn/last30days-skill:last30days-skill"
)

UPDATE=false
if [ "$1" = "--update" ]; then
  UPDATE=true
fi

echo "═══ 📚 Hermes-Omni Skill Library ═══"
echo "  Target: $REPO_DIR"
echo ""

TOTAL=0
CLONED=0
UPDATED=0
SKIPPED=0

for entry in "${REPOS[@]}"; do
  repo="${entry%%:*}"
  name="${entry##*:}"
  target="$REPO_DIR/$name"
  TOTAL=$((TOTAL + 1))

  if [ -d "$target" ]; then
    if [ "$UPDATE" = true ]; then
      echo -n "  ⏳ Updating $name... "
      cd "$target"
      git pull --ff-only --quiet 2>/dev/null && echo "✓ updated" || echo "⚠ pull failed (offline?)"
      cd - > /dev/null
      UPDATED=$((UPDATED + 1))
    else
      echo "  ⏭ $name (already exists)"
      SKIPPED=$((SKIPPED + 1))
    fi
  else
    echo -n "  ⏳ Cloning $name... "
    if git clone --depth 1 "https://github.com/$repo.git" "$target" 2>/dev/null; then
      echo "✓"
      CLONED=$((CLONED + 1))
    else
      echo "✗ failed (check network)"
    fi
  fi
done

echo ""
echo "═══ Summary ═══"
echo "  Total repos: $TOTAL"
echo "  Cloned:      $CLONED"
echo "  Updated:     $UPDATED"
echo "  Skipped:     $SKIPPED"
echo ""

# Count SKILL.md files
SKILL_COUNT=$(find "$REPO_DIR" -name "SKILL.md" 2>/dev/null | wc -l)
echo "  SKILL.md files found: $SKILL_COUNT"
echo ""

if [ "$CLONED" -gt 0 ] || [ "$UPDATED" -gt 0 ]; then
  echo "✅ Skill library ready! Agent can now search + load $SKILL_COUNT skills."
  echo ""
  echo "Usage in chat:"
  echo "  > skill_search \"debugging\""
  echo "  > skill_load \"local-agent-skills-debugging\""
else
  echo "All repos already present. Use --update to pull latest."
fi

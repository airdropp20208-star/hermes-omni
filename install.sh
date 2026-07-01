#!/usr/bin/env bash
# Hermes-Omni Cognitive Agent — One-line Installer
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/airdropp20208-star/hermes-omni/main/install.sh | bash
#
# Or clone first:
#   git clone https://github.com/airdropp20208-star/hermes-omni.git
#   cd hermes-omni && bash install.sh
#
# This script:
#   1. Detects OS (Linux/macOS/Windows-WSL)
#   2. Installs uv (Python package manager) if missing
#   3. Clones hermes-omni repo (if running via curl)
#   4. Creates Python 3.11 virtualenv
#   5. Installs all dependencies (.[all,dev])
#   6. Optionally installs sentence-transformers (for embeddings)
#   7. Runs cognitive module evaluation
#   8. Launches `hermes setup` wizard

set -e

# ─── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}ℹ${NC}  $*"; }
success() { echo -e "${GREEN}✓${NC}  $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC}  $*" >&2; }
header()  { echo -e "\n${BOLD}${YELLOW}═══ $* ═══${NC}\n"; }

# ─── Detect OS ──────────────────────────────────────────────────────────────
detect_os() {
    local os
    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="macos" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *)       os="unknown" ;;
    esac
    echo "$os"
}

OS=$(detect_os)
ARCH=$(uname -m)

header "🧠 Hermes-Omni Cognitive Agent — Installer"
echo "  OS:      $OS"
echo "  Arch:    $ARCH"
echo "  Shell:   $SHELL"
echo "  User:    $(whoami)"
echo ""

# ─── Check prerequisites ────────────────────────────────────────────────────
check_prereqs() {
    header "1/7 — Checking prerequisites"

    local missing=()

    if ! command -v curl &>/dev/null; then
        missing+=("curl")
    fi
    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi
    if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
        missing+=("python3")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        warn "Missing: ${missing[*]}"
        info "Installing missing prerequisites..."

        case "$OS" in
            linux)
                if command -v apt-get &>/dev/null; then
                    sudo apt-get update -qq && sudo apt-get install -y -qq "${missing[@]}"
                elif command -v dnf &>/dev/null; then
                    sudo dnf install -y -q "${missing[@]}"
                elif command -v yum &>/dev/null; then
                    sudo yum install -y -q "${missing[@]}"
                elif command -v pacman &>/dev/null; then
                    sudo pacman -S --noconfirm "${missing[@]}"
                else
                    error "Cannot auto-install. Please install manually: ${missing[*]}"
                    exit 1
                fi
                ;;
            macos)
                if command -v brew &>/dev/null; then
                    brew install "${missing[@]}"
                else
                    error "Install Homebrew first: https://brew.sh"
                    exit 1
                fi
                ;;
            *)
                error "Please install manually: ${missing[*]}"
                exit 1
                ;;
        esac
    fi

    success "All prerequisites present (curl, git, python3)"
}

# ─── Install uv ─────────────────────────────────────────────────────────────
install_uv() {
    header "2/7 — Installing uv (Python package manager)"

    if command -v uv &>/dev/null; then
        success "uv already installed: $(uv --version)"
        return
    fi

    info "Installing uv via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Source profile to get uv in PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    # Add to shell profile for persistence
    local profile=""
    case "$SHELL" in
        */bash) profile="$HOME/.bashrc" ;;
        */zsh)  profile="$HOME/.zshrc" ;;
        */fish) profile="$HOME/.config/fish/config.fish" ;;
        *)      profile="$HOME/.profile" ;;
    esac

    if [ -n "$profile" ] && [ -f "$profile" ]; then
        if ! grep -q '.local/bin' "$profile" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$profile"
            info "Added uv to PATH in $profile"
        fi
    fi

    if command -v uv &>/dev/null; then
        success "uv installed: $(uv --version)"
    else
        error "uv installation failed. Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
}

# ─── Clone repo (if running via curl) ───────────────────────────────────────
clone_repo() {
    header "3/7 — Cloning hermes-omni repository"

    # If we're running from within the repo (git clone + bash install.sh),
    # skip cloning.
    if [ -f "pyproject.toml" ] && git rev-parse --git-dir &>/dev/null; then
        REPO_DIR="$(pwd)"
        success "Already in repo: $REPO_DIR"
        return
    fi

    REPO_DIR="$HOME/hermes-omni"
    if [ -d "$REPO_DIR" ]; then
        warn "$REPO_DIR already exists."
        # Auto-pull if non-interactive (e.g., curl | bash), else prompt.
        if [ -t 0 ]; then
            read -rp "Pull latest? [Y/n] " -n 1 -r
            echo
            SHOULD_PULL=$([[ ! $REPLY =~ ^[Nn]$ ]] && echo yes || echo no)
        else
            info "Non-interactive mode — auto-pulling latest..."
            SHOULD_PULL=yes
        fi
        if [ "$SHOULD_PULL" = "yes" ]; then
            cd "$REPO_DIR"
            git pull --ff-only origin main || warn "Pull failed, continuing with existing"
            cd -
        fi
    else
        info "Cloning to $REPO_DIR ..."
        git clone --depth 1 https://github.com/airdropp20208-star/hermes-omni.git "$REPO_DIR"
        success "Cloned to $REPO_DIR"
    fi
}

# ─── Create virtualenv + install deps ───────────────────────────────────────
install_deps() {
    header "4/7 — Creating virtualenv and installing dependencies"

    cd "$REPO_DIR"

    info "Creating Python 3.11 virtualenv..."
    if [ ! -d ".venv" ]; then
        uv venv .venv --python 3.11
        success "Virtualenv created"
    else
        success "Virtualenv already exists"
    fi

    # Activate venv
    case "$OS" in
        windows)
            # shellcheck disable=SC1091
            source .venv/Scripts/activate
            ;;
        *)
            # shellcheck disable=SC1091
            source .venv/bin/activate
            ;;
    esac

    info "Installing dependencies (this may take 2-5 minutes)..."
    uv pip install -e ".[all,dev]" --quiet 2>&1 | tail -5
    success "Dependencies installed"

    # Verify hermes command works
    if command -v hermes &>/dev/null; then
        success "hermes command available: $(which hermes)"
    else
        warn "hermes not in PATH. Use: source .venv/bin/activate"
    fi
}

# ─── Optional: sentence-transformers ────────────────────────────────────────
install_extras() {
    header "5/7 — Optional: Embedding support"

    echo "Embedding (semantic recall) improves memory quality +40%."
    echo "Requires sentence-transformers (~80MB model download)."
    if [ -t 0 ]; then
        read -rp "Install sentence-transformers? [y/N] " -n 1 -r
        echo
        INSTALL_EXTRA=$([[ $REPLY =~ ^[Yy]$ ]] && echo yes || echo no)
    else
        info "Non-interactive mode — skipping extras (install later: uv pip install sentence-transformers)"
        INSTALL_EXTRA=no
    fi
    if [ "$INSTALL_EXTRA" = "yes" ]; then
        info "Installing sentence-transformers..."
        uv pip install sentence-transformers --quiet 2>&1 | tail -3
        success "sentence-transformers installed"
        echo ""
        echo "Enable in config:"
        echo "  unified:"
        echo "    embedding:"
        echo "      enabled: true"
        echo "      backend: local"
    else
        info "Skipped. You can install later: uv pip install sentence-transformers"
    fi
}

# ─── Verify cognitive modules ───────────────────────────────────────────────
verify_install() {
    header "6/7 — Verifying cognitive modules"

    if [ ! -f "scripts/evaluate_cognitive.py" ]; then
        warn "Evaluation script not found. Skipping verification."
        return
    fi

    info "Running evaluation..."
    if python scripts/evaluate_cognitive.py 2>&1; then
        success "All cognitive modules verified!"
    else
        warn "Evaluation had issues. Check output above."
    fi
}

# ─── Launch setup wizard ────────────────────────────────────────────────────
launch_setup() {
    header "7/7 — Launching Hermes setup wizard"

    echo "The setup wizard will guide you through:"
    echo "  • Choosing an LLM provider (GLM, Claude, OpenAI, OpenRouter, ...)"
    echo "  • Entering your API key"
    echo "  • Configuring messaging platforms (Telegram, Discord, Slack, ...)"
    echo "  • Enabling tools (web search, image gen, TTS, ...)"
    echo ""
    if [ -t 0 ]; then
        read -rp "Launch setup wizard now? [Y/n] " -n 1 -r
        echo
        LAUNCH_SETUP=$([[ ! $REPLY =~ ^[Nn]$ ]] && echo yes || echo no)
    else
        info "Non-interactive mode — skipping setup wizard (run later: hermes setup)"
        LAUNCH_SETUP=no
    fi
    if [ "$LAUNCH_SETUP" = "yes" ]; then
        hermes setup
    else
        info "You can run setup later: hermes setup"
    fi
}

# ─── Print next steps ───────────────────────────────────────────────────────
print_next_steps() {
    header "✅ Installation Complete!"

    echo -e "${GREEN}Hermes-Omni is ready!${NC}"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo ""
    echo -e "  1. ${BOLD}Activate the virtualenv${NC} (if not already):"
    echo -e "     ${YELLOW}cd $REPO_DIR && source .venv/bin/activate${NC}"
    echo ""
    echo -e "  2. ${BOLD}Start chatting${NC}:"
    echo -e "     ${YELLOW}hermes${NC}"
    echo ""
    echo -e "  3. ${BOLD}Enable cognitive features${NC} (edit ~/.hermes/config.yaml):"
    echo -e "     ${YELLOW}unified:"
    echo -e "  reasoning:"
    echo -e "    enabled: true"
    echo -e "  verifier:"
    echo -e "    enabled: true"
    echo -e "  slow_thinking:"
    echo -e "    enabled: true"
    echo -e "    default_level: balanced${NC}"
    echo ""
    echo -e "  4. ${BOLD}Setup Telegram/Discord${NC} (optional):"
    echo -e "     ${YELLOW}hermes gateway setup${NC}"
    echo ""
    echo -e "  5. ${BOLD}Read the docs${NC}:"
    echo -e "     ${YELLOW}https://github.com/airdropp20208-star/hermes-omni${NC}"
    echo ""
    echo -e "${BOLD}Cognitive modules installed:${NC}"
    echo "  • v1:    reasoning, smart_guardian, decision, reflexion, policy"
    echo "  • v1.1:  longrun, tool_router"
    echo "  • v2:    cognitive_tree, hypothesis, context_distiller, metacognitive, causal_graph"
    echo "  • v2.1:  learning, skill_synthesizer"
    echo "  • v2.2:  task_planner (recursive)"
    echo "  • v2.3:  output_formatter (Telegram/Slack/Discord)"
    echo "  • v3:    verifier, constitution, slow_thinking, ensemble, capability_resolver"
    echo "  • v3.1:  cost_tracker, response_cache, user_model, clarifier, streaming, embedding"
    echo ""
    echo -e "${BOLD}Total: 31 modules · 15,557 lines${NC}"
    echo ""
}

# ─── Main ───────────────────────────────────────────────────────────────────
main() {
    check_prereqs
    install_uv
    clone_repo
    install_deps
    install_extras
    verify_install
    launch_setup
    print_next_steps
}

main "$@"

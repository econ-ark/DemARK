#!/bin/bash
# setup_conda.sh
# Simple script to set up conda initialization for the DemARK scripts

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

main() {
    log_info "Setting up conda initialization for DemARK scripts"
    echo
    
    # Detect current shell
    local current_shell=$(basename "$SHELL")
    log_info "Detected shell: $current_shell"
    
    # Check if conda is available
    if ! command -v conda >/dev/null 2>&1; then
        log_error "Conda not found in PATH"
        log_error "Please install conda/miniconda/mamba first"
        exit 1
    fi
    
    log_success "Conda found at: $(which conda)"
    
    # Initialize conda for the current shell
    log_info "Initializing conda for $current_shell..."
    conda init "$current_shell"
    
    # Check if direnv is available
    if command -v direnv >/dev/null 2>&1; then
        log_success "direnv found - automatic environment switching will be available"
        
        # Add direnv hook if not already present
        local shell_rc=""
        case "$current_shell" in
            bash) shell_rc="$HOME/.bashrc" ;;
            zsh) shell_rc="$HOME/.zshrc" ;;
            *) shell_rc="$HOME/.${current_shell}rc" ;;
        esac
        
        if [[ -f "$shell_rc" ]] && ! grep -q "direnv hook" "$shell_rc"; then
            log_info "Adding direnv hook to $shell_rc"
            echo 'eval "$(direnv hook '"$current_shell"')"' >> "$shell_rc"
        fi
    else
        log_warning "direnv not found - automatic environment switching won't work"
        log_info "To install direnv: brew install direnv"
    fi
    
    echo
    log_success "Conda initialization complete"
    log_warning "IMPORTANT: You must restart your shell or run:"
    log_warning "  source ~/.${current_shell}rc"
    log_warning "Then you can use the DemARK environment scripts."
    echo
    log_info "After restarting your shell, you can:"
    log_info "  • Run ./last-working-ci-before-0p16-preserve.sh to create both environments"
    log_info "  • Use conda activate to switch between environments"
    log_info "  • Use direnv for automatic environment switching (if installed)"
}

main "$@" 
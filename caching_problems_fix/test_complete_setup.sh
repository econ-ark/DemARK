#!/bin/bash
# test_complete_setup.sh
# Complete automated setup and testing script for DemARK CI caching issue reproduction

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}${BOLD}[STEP]${NC} $1"; }

# Global variables for directory names
HISTORICAL_DIR=""
CURRENT_DIR=""

main() {
    echo -e "${BOLD}${CYAN}"
    echo "============================================================"
    echo "  DemARK CI Caching Issue - Complete Setup & Test Script"
    echo "============================================================"
    echo -e "${NC}"
    
    log_step "1/6: Initial Setup and Conda Configuration"
    setup_conda_environment
    
    log_step "2/6: Cleaning Up Existing Environments"
    cleanup_existing_environments
    
    log_step "3/6: Creating Working Environments"
    create_environments
    
    log_step "4/6: Testing Environment Creation"
    test_environment_creation
    
    log_step "5/6: Setting Up direnv (if available)"
    setup_direnv
    
    log_step "6/6: Providing Verification Instructions"
    provide_verification_instructions
    
    echo
    echo -e "${BOLD}${GREEN}🎉 SETUP COMPLETE! Follow the verification instructions above.${NC}"
}

setup_conda_environment() {
    log_info "Setting up conda initialization..."
    
    # Check if conda is available
    if ! command -v conda >/dev/null 2>&1; then
        log_error "Conda not found in PATH"
        log_error "Please install conda/miniconda/mamba first"
        exit 1
    fi
    
    # Detect shell and initialize conda
    local current_shell=$(basename "$SHELL")
    log_info "Detected shell: $current_shell"
    
    # Initialize conda (this is idempotent)
    conda init "$current_shell" >/dev/null 2>&1 || true
    
    # Source conda setup if available
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    
    log_success "Conda configuration updated"
}

cleanup_existing_environments() {
    log_info "Removing conflicting environments..."
    
    # Remove conflicting conda environments
    conda env remove -n DemARK --yes 2>/dev/null || true
    conda env remove -n DemARK_historical_working --yes 2>/dev/null || true
    conda env remove -n DemARK_current_fixed --yes 2>/dev/null || true
    
    # Remove existing worktrees
    for dir in ../DemARK_historical_* ../DemARK_current_*; do
        if [[ -d "$dir" ]]; then
            log_info "Removing existing worktree: $dir"
            git worktree remove "$dir" --force 2>/dev/null || true
        fi
    done
    
    log_success "Cleanup completed"
}

create_environments() {
    log_info "Running the main environment creation script..."
    
    # Run the main script and capture output
    if ./caching_problems_fix/last-working-ci-before-0p16-preserve.sh; then
        log_success "Environment creation script completed successfully"
        
        # Find the created directories
        HISTORICAL_DIR=$(ls -1d ../DemARK_historical_* 2>/dev/null | head -1 || echo "")
        CURRENT_DIR=$(ls -1d ../DemARK_current_* 2>/dev/null | head -1 || echo "")
        
        if [[ -z "$HISTORICAL_DIR" ]] || [[ -z "$CURRENT_DIR" ]]; then
            log_error "Could not find created directories"
            exit 1
        fi
        
        log_success "Created directories:"
        log_success "  Historical: $HISTORICAL_DIR"
        log_success "  Current: $CURRENT_DIR"
    else
        log_error "Environment creation script failed"
        exit 1
    fi
}

test_environment_creation() {
    log_info "Testing that environments were created correctly..."
    
    # Test historical environment
    if conda env list | grep -q "DemARK_historical_working"; then
        log_success "✅ DemARK_historical_working environment exists"
    else
        log_error "❌ DemARK_historical_working environment not found"
        exit 1
    fi
    
    # Test current environment
    if conda env list | grep -q "DemARK_current_fixed"; then
        log_success "✅ DemARK_current_fixed environment exists"
    else
        log_error "❌ DemARK_current_fixed environment not found"
        exit 1
    fi
    
    # Test directories exist
    if [[ -d "$HISTORICAL_DIR" ]]; then
        log_success "✅ Historical worktree directory exists"
    else
        log_error "❌ Historical worktree directory not found"
        exit 1
    fi
    
    if [[ -d "$CURRENT_DIR" ]]; then
        log_success "✅ Current worktree directory exists"
    else
        log_error "❌ Current worktree directory not found"
        exit 1
    fi
}

setup_direnv() {
    if command -v direnv >/dev/null 2>&1; then
        log_info "Setting up direnv auto-activation..."
        
        # Allow direnv in both directories
        (cd "$HISTORICAL_DIR" && direnv allow 2>/dev/null) || true
        (cd "$CURRENT_DIR" && direnv allow 2>/dev/null) || true
        
        log_success "direnv configured for automatic environment switching"
    else
        log_warning "direnv not found - automatic environment switching won't work"
        log_info "To install: brew install direnv"
    fi
}

provide_verification_instructions() {
    echo
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}  VERIFICATION INSTRUCTIONS${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo
    
    echo -e "${BOLD}📋 STEP 1: Restart Your Shell${NC}"
    echo "First, restart your shell to ensure conda is properly initialized:"
    echo -e "${YELLOW}exec bash${NC}"
    echo
    
    echo -e "${BOLD}📋 STEP 2: Test Historical Environment (Nov 2023 - Cached Version)${NC}"
    echo "This tests the environment that was actually cached in GitHub CI:"
    echo
    echo -e "${CYAN}# Navigate to historical directory${NC}"
    echo "cd $HISTORICAL_DIR"
    echo
    echo -e "${CYAN}# Activate historical environment${NC}"
    echo "conda activate DemARK_historical_working"
    echo
    echo -e "${CYAN}# Test old HARK imports (should work)${NC}"
    echo 'python -c "from HARK.datasets import load_SCF_wealth_weights; print(\"✅ Historical HARK.datasets import works\"); print(\"HARK version:\", __import__(\"HARK\").__version__)"'
    echo
    echo -e "${CYAN}# Run notebook tests (should pass)${NC}"
    echo "python -m pytest --nbval-lax --nbval-cell-timeout=12000 \\"
    echo "  --ignore=notebooks/Chinese-Growth.ipynb \\"
    echo "  --ignore=notebooks/Harmenberg-Aggregation.ipynb \\"
    echo "  notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb \\"
    echo "  notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb \\"
    echo "  -v"
    echo
    echo -e "${GREEN}Expected: HARK version 0.13.0, old imports work, tests pass${NC}"
    echo
    
    echo -e "${BOLD}📋 STEP 3: Test Current Environment (Fixed Version)${NC}"
    echo "This tests the properly updated environment with new imports:"
    echo
    echo -e "${CYAN}# Navigate to current directory${NC}"
    echo "cd $CURRENT_DIR"
    echo
    echo -e "${CYAN}# Activate current environment${NC}"
    echo "conda activate DemARK_current_fixed"
    echo
    echo -e "${CYAN}# Test new HARK imports (should work)${NC}"
    echo 'python -c "from HARK.Calibration import load_SCF_wealth_weights; print(\"✅ Current HARK.Calibration import works\"); print(\"HARK version:\", __import__(\"HARK\").__version__)"'
    echo
    echo -e "${CYAN}# Run notebook tests (should pass)${NC}"
    echo "python -m pytest --nbval-lax --nbval-cell-timeout=12000 \\"
    echo "  --ignore=notebooks/Chinese-Growth.ipynb \\"
    echo "  --ignore=notebooks/Harmenberg-Aggregation.ipynb \\"
    echo "  notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb \\"
    echo "  notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb \\"
    echo "  -v"
    echo
    echo -e "${GREEN}Expected: HARK version 0.16.0+, new imports work, tests pass${NC}"
    echo
    
    if command -v direnv >/dev/null 2>&1; then
        echo -e "${BOLD}📋 STEP 4: Test Automatic Environment Switching (Optional)${NC}"
        echo "Test that environments auto-activate when you change directories:"
        echo
        echo -e "${CYAN}# Switch between directories and watch environment change${NC}"
        echo "cd $HISTORICAL_DIR"
        echo "# Should show: direnv loading and activate DemARK_historical_working"
        echo
        echo "cd $CURRENT_DIR"
        echo "# Should show: direnv loading and activate DemARK_current_fixed"
        echo
    fi
    
    echo -e "${BOLD}📋 CLEANUP (when done testing):${NC}"
    echo -e "${CYAN}# Remove worktrees${NC}"
    echo "git worktree remove $HISTORICAL_DIR"
    echo "git worktree remove $CURRENT_DIR"
    echo
    echo -e "${CYAN}# Remove environments${NC}"
    echo "conda env remove -n DemARK_historical_working"
    echo "conda env remove -n DemARK_current_fixed"
    echo
    
    echo -e "${BOLD}🎯 What This Proves:${NC}"
    echo "• Historical environment reproduces the 'successful' cached CI state"
    echo "• Current environment shows the proper fix with updated imports"
    echo "• The CI was successful only because of 11-month-old cached environment"
    echo "• Fresh environments failed because HARK changed but DemARK didn't update"
    echo
}

# Run main function
main "$@" 
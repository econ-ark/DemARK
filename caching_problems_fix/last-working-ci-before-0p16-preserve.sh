#!/usr/bin/env bash
# last-working-ci-before-0p16-preserve.sh
# Purpose: Create TWO working environments for comparison
# 
# This script creates:
# 1. Historical working state (Nov 2023) - notebooks with old imports + HARK with old structure
# 2. Current fixed state (main branch) - notebooks with new imports + HARK with new structure

set -euo pipefail

# Configuration
ENV_NAME_HISTORICAL="DemARK_historical_working"
ENV_NAME_CURRENT="DemARK_current_fixed"
TIMESTAMP=$(date +%s)
WORKTREE_DIR_HISTORICAL="../DemARK_historical_$TIMESTAMP"
WORKTREE_DIR_CURRENT="../DemARK_current_$TIMESTAMP"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Function to get historical commit (when cache was created)
get_historical_commit() {
    # Use the commit when the cache was actually created (Nov 29, 2023)
    # This is when notebooks had old imports AND HARK had old structure - genuinely working
    local commit="6d0aa34"  # Last commit before HARK breaking changes where everything truly worked
    echo "$commit"
    return 0
}

# Function to get current main branch commit
get_current_commit() {
    # Get the current main branch commit (with updated notebooks)
    local commit=$(git rev-parse main)
    echo "$commit"
    return 0
}

# Function to get HARK commits
get_hark_historical_commit() {
    # HARK commit that hark@master actually resolved to on Nov 29, 2023 (when cache was created)
    # This is what was ACTUALLY cached, not just "before breaking changes"
    echo "1ad4731d"
    return 0
}

get_hark_current_commit() {
    # Current HARK master
    echo "master"
    return 0
}



# Function to validate discovered commits
validate_commits() {
    local demark_commit="$1"
    local hark_commit="$2"
    
    log_info "Validating discovered commits..."
    
    # Verify DemARK commit exists
    if ! git cat-file -e "$demark_commit" 2>/dev/null; then
        log_error "DemARK commit $demark_commit not found"
        return 1
    fi
    
    # Check if HARK repository exists
    if [[ ! -d "../HARK" ]]; then
        log_error "HARK repository not found at ../HARK"
        log_error "Please ensure the HARK repository is cloned adjacent to the DemARK directory"
        return 1
    fi
    
    # Verify HARK target commit exists
    if ! (cd ../HARK && git cat-file -e "$hark_commit" 2>/dev/null); then
        log_error "HARK commit $hark_commit not found in ../HARK"
        return 1
    fi
    
    log_success "Commits validated successfully"
    return 0
}

# Function to create environment with specific HARK version
create_environment() {
    local env_name="$1"
    local hark_commit="$2"
    local description="$3"
    
    log_info "Creating environment: $env_name ($description)"
    
    # Create environment.yml with specified HARK commit
    cat > binder/environment.yml << EOF
name: $env_name
channels:
  - conda-forge
dependencies:
  - python=3.10
  - matplotlib
  - numpy
  - ipywidgets
  - seaborn
  - scipy
  - pandas
  - pandas-datareader
  - statsmodels
  - linearmodels
  - tqdm
  - nbval
  - pip
  - pip:
    - git+https://github.com/econ-ark/hark@$hark_commit
EOF
    
    # Remove existing environment if it exists
    conda env remove -n "$env_name" --yes 2>/dev/null || true
    
    # Create the environment
    mamba env create -f binder/environment.yml
    
    # Create convenience scripts for automatic environment activation
    create_convenience_scripts "$env_name"
    
    log_success "Created environment: $env_name with convenience scripts"
}

# Function to create convenience scripts for automatic environment activation
create_convenience_scripts() {
    local env_name="$1"
    
    # Create .envrc for direnv users
    echo "conda activate $env_name" > .envrc
    
    # Create test runner script with auto-activation
    cat > run_tests.sh << EOF
#!/bin/bash
# Auto-activate environment and run tests
set -euo pipefail

# Check if conda is properly initialized
if ! command -v conda activate >/dev/null 2>&1; then
    echo "⚠️  Conda not properly initialized. Run these commands:"
    echo "   conda init \$(basename \$SHELL)"
    echo "   source ~/.bashrc  # or restart your shell"
    echo "   Then try again."
    exit 1
fi

echo "🔧 Activating $env_name environment..."
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

echo "🧪 Running tests in $env_name environment..."
python -m pytest --nbval-lax --nbval-cell-timeout=12000 \\
  --ignore=notebooks/Chinese-Growth.ipynb \\
  --ignore=notebooks/Harmenberg-Aggregation.ipynb \\
  notebooks/ -v
EOF
    chmod +x run_tests.sh
    
    # Create simple activation script
    cat > activate_env.sh << EOF
#!/bin/bash
# Check if conda is properly initialized
if ! command -v conda activate >/dev/null 2>&1; then
    echo "⚠️  Conda not properly initialized. Run these commands:"
    echo "   conda init \$(basename \$SHELL)"
    echo "   source ~/.bashrc  # or restart your shell"
    echo "   Then try again."
    exit 1
fi

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name
echo "🔧 Activated environment: $env_name"
echo "📋 Run tests with: python -m pytest --nbval-lax notebooks/ -v"
EOF
    chmod +x activate_env.sh
}

# Main execution
main() {
    log_info "Creating TWO DemARK environments for comparison"
    echo
    log_info "1. Historical working state (Nov 2023): notebooks with old imports + HARK with old structure"
    log_info "2. Current fixed state (main branch): notebooks with new imports + HARK with new structure"
    echo
    
    # Verify we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Get commits
    local historical_commit=$(get_historical_commit)
    local current_commit=$(get_current_commit)
    local hark_historical=$(get_hark_historical_commit)
    local hark_current=$(get_hark_current_commit)
    
    log_info "Commits to use:"
    log_info "  • Historical DemARK: $historical_commit (Nov 29, 2023 - when cache was created)"
    log_info "  • Current DemARK: $current_commit (main branch with fixed imports)"
    log_info "  • Historical HARK: $hark_historical (what hark@master resolved to on Nov 29, 2023)"
    log_info "  • Current HARK: $hark_current (current master)"
    echo
    
    # Validate commits
    if ! validate_commits "$historical_commit" "$hark_historical"; then
        exit 1
    fi
    
    # Create historical worktree
    log_info "Creating historical worktree at $WORKTREE_DIR_HISTORICAL"
    git worktree add "$WORKTREE_DIR_HISTORICAL" "$historical_commit"
    
    # Create current worktree
    log_info "Creating current worktree at $WORKTREE_DIR_CURRENT"
    git worktree add "$WORKTREE_DIR_CURRENT" "$current_commit"
    
    # Set up historical environment
    log_info "Setting up HISTORICAL working environment..."
    cd "$WORKTREE_DIR_HISTORICAL"
    create_environment "$ENV_NAME_HISTORICAL" "$hark_historical" "historical HARK before breaking changes"
    
    # Set up current environment
    log_info "Setting up CURRENT fixed environment..."
    cd "$WORKTREE_DIR_CURRENT"
    create_environment "$ENV_NAME_CURRENT" "$hark_current" "current HARK master"
    
    # Return to original directory (preserving scripts)
    cd "$SCRIPT_DIR/.."
    
    # Final success message
    echo
    log_success "🎉 SUCCESS: Both environments created with preserved worktrees"
    echo
    log_info "=== HISTORICAL WORKING STATE (Nov 2023) ==="
    log_info "  • Environment: $ENV_NAME_HISTORICAL"
    log_info "  • DemARK: $historical_commit (notebooks with old imports: HARK.datasets)"
    log_info "  • HARK: $hark_historical (what hark@master resolved to when cache was created)"
    log_info "  • Worktree: $WORKTREE_DIR_HISTORICAL"
    log_info "  • Test options:"
    log_info "    cd $WORKTREE_DIR_HISTORICAL"
    log_info "    ./run_tests.sh                    # Auto-activates environment and runs tests"
    log_info "    # OR manually:"
    log_info "    ./activate_env.sh                 # Just activate environment"
    log_info "    conda activate $ENV_NAME_HISTORICAL"
    echo
    log_info "=== CURRENT FIXED STATE (main branch) ==="
    log_info "  • Environment: $ENV_NAME_CURRENT"
    log_info "  • DemARK: $current_commit (notebooks with new imports: HARK.Calibration)"
    log_info "  • HARK: $hark_current (current master)"
    log_info "  • Worktree: $WORKTREE_DIR_CURRENT"
    log_info "  • Test options:"
    log_info "    cd $WORKTREE_DIR_CURRENT"
    log_info "    ./run_tests.sh                    # Auto-activates environment and runs tests"
    log_info "    # OR manually:"
    log_info "    ./activate_env.sh                 # Just activate environment"
    log_info "    conda activate $ENV_NAME_CURRENT"
    echo
    log_warning "IMPORTANT: Both worktrees preserved for manual testing"
    echo
    log_info "🎉 DIRENV INTEGRATION:"
    if command -v direnv >/dev/null 2>&1; then
        log_success "✅ direnv is installed - environments auto-activate when you cd"
        log_info "  • To enable auto-activation, run these commands:"
        log_info "    conda init \$(basename \$SHELL)"
        log_info "    source ~/.bashrc  # or restart your shell"
        log_info "    cd $WORKTREE_DIR_HISTORICAL && direnv allow"
        log_info "    cd $WORKTREE_DIR_CURRENT && direnv allow"
        log_info "  • Then cd into any worktree and the environment activates automatically"
    else
        log_info "💡 For automatic environment activation, install direnv:"
        log_info "  brew install direnv"
        log_info "  echo 'eval \"\$(direnv hook bash)\"' >> ~/.bashrc"
        log_info "  conda init \$(basename \$SHELL)"
        log_info "  source ~/.bashrc  # or restart your shell"
        log_info "  cd $WORKTREE_DIR_HISTORICAL && direnv allow"
        log_info "  cd $WORKTREE_DIR_CURRENT && direnv allow"
    fi
    echo
    log_warning "To clean up later:"
    log_warning "  git worktree remove $WORKTREE_DIR_HISTORICAL"
    log_warning "  git worktree remove $WORKTREE_DIR_CURRENT"
    log_warning "  conda env remove -n $ENV_NAME_HISTORICAL"
    log_warning "  conda env remove -n $ENV_NAME_CURRENT"
}

# Run main function
main "$@" 
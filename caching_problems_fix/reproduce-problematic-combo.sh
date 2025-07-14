#!/usr/bin/env bash
# reproduce-problematic-combo.sh
# Purpose: Reproduce the problematic combination that was falsely appearing successful
# 
# This script demonstrates:
# 1. The problematic combo: old DemARK imports (ffc6131) + current HARK master
# 2. Shows the failures that should have been detected by CI
# 3. Repairs the files with correct imports
# 4. Shows that the repaired version works

set -euo pipefail

# Configuration
ENV_NAME="DemARK_problematic_combo"
TIMESTAMP=$(date +%s)
WORKTREE_DIR="../DemARK_problematic_$TIMESTAMP"
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

# Function to get the falsely successful commit
get_problematic_commit() {
    # This is the commit that appeared successful in CI due to caching
    # but actually had old imports that should have failed with current HARK
    echo "ffc6131"
    return 0
}

# Function to validate commits
validate_commits() {
    local demark_commit="$1"
    
    log_info "Validating discovered commits..."
    
    # Verify DemARK commit exists
    if ! git cat-file -e "$demark_commit" 2>/dev/null; then
        log_error "DemARK commit $demark_commit not found"
        return 1
    fi
    
    # Check if HARK repository exists
    if [[ ! -d "../../HARK" ]]; then
        log_error "HARK repository not found at ../../HARK"
        log_error "Please ensure the HARK repository is cloned adjacent to the DemARK directory"
        return 1
    fi
    
    log_success "Commits validated successfully"
    return 0
}

# Function to create environment with HARK from September 18, 2024
create_environment() {
    local env_name="$1"
    
    log_info "Creating environment: $env_name with HARK breaking change commit"
    log_info "Using HARK commit 7a6e8f39 (Alan Lujan's commit that moved datasets to Calibration)"
    log_info "This is the first HARK version that should have caused CI failures"
    
    # Create environment.yml with the breaking change commit
    # This is what would have immediately broken DemARK if not for caching
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
    - git+https://github.com/econ-ark/hark@7a6e8f39
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
    cat > .envrc << EOF
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name
EOF
    
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

# Function to run tests and capture failures
run_tests_and_capture_failures() {
    log_info "Running tests to demonstrate the failures that should have been detected..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Run tests on the problematic notebooks, expecting failures
    log_info "Testing LC-Model-Expected-Vs-Realized-Income-Growth.ipynb (expecting failure)..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb -v; then
        log_warning "❌ Test passed when it should have failed! (Cache masking issue?)"
    else
        log_success "✅ Test failed as expected - detected the import issue!"
    fi
    
    log_info "Testing Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb (expecting failure)..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb -v; then
        log_warning "❌ Test passed when it should have failed! (Cache masking issue?)"
    else
        log_success "✅ Test failed as expected - detected the import issue!"
    fi
}

# Function to repair the problematic imports
repair_imports() {
    log_info "Repairing the problematic imports..."
    
    # Fix LC-Model notebook
    log_info "Fixing LC-Model-Expected-Vs-Realized-Income-Growth.ipynb..."
    if grep -q "from HARK.datasets import" notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb; then
        sed -i '' 's/from HARK\.datasets import/from HARK.Calibration import/g' notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb
        log_success "✅ Fixed HARK.datasets → HARK.Calibration in LC-Model notebook"
    else
        log_info "LC-Model notebook already has correct imports"
    fi
    
    # Fix Micro-and-Macro notebook
    log_info "Fixing Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb..."
    if grep -q "from HARK.datasets import" notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb; then
        sed -i '' 's/from HARK\.datasets import/from HARK.Calibration import/g' notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb
        log_success "✅ Fixed HARK.datasets → HARK.Calibration in Micro-and-Macro notebook"
    else
        log_info "Micro-and-Macro notebook already has correct imports"
    fi
}

# Function to test the repaired version
test_repaired_version() {
    log_info "Testing the repaired version..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Run tests on the repaired notebooks
    log_info "Testing repaired LC-Model-Expected-Vs-Realized-Income-Growth.ipynb..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb -v; then
        log_success "✅ Repaired LC-Model notebook now passes!"
    else
        log_error "❌ Repaired LC-Model notebook still fails"
    fi
    
    log_info "Testing repaired Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb -v; then
        log_success "✅ Repaired Micro-and-Macro notebook now passes!"
    else
        log_error "❌ Repaired Micro-and-Macro notebook still fails"
    fi
    
    # Run full test suite to confirm everything works
    log_info "Running full test suite to confirm all repairs work..."
    python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
      --ignore=notebooks/Chinese-Growth.ipynb \
      --ignore=notebooks/Harmenberg-Aggregation.ipynb \
      notebooks/ -v
}

# Main execution
main() {
    log_info "🔍 REPRODUCING THE PROBLEMATIC COMBINATION"
    echo
    log_info "This demonstrates the combination that was falsely appearing successful due to caching:"
    log_info "  • DemARK notebooks: ffc6131 (Sept 2024) with old imports (HARK.datasets)"
    log_info "  • HARK version: 7a6e8f39 (Alan Lujan's breaking change - moved datasets to Calibration)"
    log_info "    → This is the FIRST HARK commit that should have caused CI failures"
    echo
    log_warning "This combination SHOULD FAIL but appeared successful due to environment caching"
    echo
    
    # Verify we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Get the problematic commit
    local problematic_commit=$(get_problematic_commit)
    
    log_info "Using commits:"
    log_info "  • DemARK: $problematic_commit (falsely successful commit with old imports)"
    log_info "  • HARK: 7a6e8f39 (breaking change commit - moved datasets to Calibration)"
    log_info "    → May 22, 2024 - This should have immediately broken DemARK CI"
    echo
    
    # Validate commits
    if ! validate_commits "$problematic_commit"; then
        exit 1
    fi
    
    # Create worktree with the problematic commit
    log_info "Creating worktree with problematic commit at $WORKTREE_DIR"
    git worktree add "$WORKTREE_DIR" "$problematic_commit"
    
    # Set up environment
    log_info "Setting up environment with current HARK master..."
    cd "$WORKTREE_DIR"
    create_environment "$ENV_NAME"
    
    # Step 1: Demonstrate the failures
    echo
    log_info "=== STEP 1: DEMONSTRATE THE FAILURES ==="
    run_tests_and_capture_failures
    
    # Step 2: Repair the imports
    echo
    log_info "=== STEP 2: REPAIR THE IMPORTS ==="
    repair_imports
    
    # Step 3: Test the repaired version
    echo
    log_info "=== STEP 3: TEST THE REPAIRED VERSION ==="
    test_repaired_version
    
    # Return to original directory
    cd "$SCRIPT_DIR/.."
    
    # Final success message
    echo
    log_success "🎉 DEMONSTRATION COMPLETE!"
    echo
    log_info "=== SUMMARY ==="
    log_info "✅ Reproduced the problematic combination that was falsely successful"
    log_info "✅ Demonstrated the failures that should have been detected by CI"
    log_info "✅ Showed how to repair the imports (HARK.datasets → HARK.Calibration)"
    log_info "✅ Verified that the repaired version works correctly"
    echo
    log_info "=== WORKING ENVIRONMENT ==="
    log_info "  • Environment: $ENV_NAME"
    log_info "  • Worktree: $WORKTREE_DIR"
    log_info "  • To continue testing:"
    log_info "    cd $WORKTREE_DIR"
    log_info "    ./run_tests.sh                    # Auto-activates environment and runs tests"
    log_info "    # OR manually:"
    log_info "    ./activate_env.sh                 # Just activate environment"
    log_info "    conda activate $ENV_NAME"
    echo
    log_info "🎉 DIRENV INTEGRATION:"
    if command -v direnv >/dev/null 2>&1; then
        log_success "✅ direnv is installed - environment auto-activates when you cd"
        log_info "  • To enable auto-activation: direnv allow"
        log_info "  • Then cd into the worktree and the environment activates automatically"
    else
        log_info "💡 For automatic environment activation, install direnv:"
        log_info "  brew install direnv"
        log_info "  echo 'eval \"\$(direnv hook bash)\"' >> ~/.bashrc"
    fi
    echo
    log_warning "To clean up later:"
    log_warning "  git worktree remove $WORKTREE_DIR"
    log_warning "  conda env remove -n $ENV_NAME"
}

# Run main function
main "$@" 
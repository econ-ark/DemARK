#!/usr/bin/env bash
# create_repaired_branch.sh
# Purpose: Create a permanent git branch with the repaired notebooks and environment setup
#
# This script creates a new branch 'repaired-scripts-after-breaking-change' that contains:
# 1. The fixed notebooks that work with HARK after the breaking changes
# 2. The environment configuration that demonstrates the problem and solution
# 3. All the demonstration scripts for reproducing the issue

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

# Configuration
BRANCH_NAME="repaired-scripts-after-breaking-change"
BASE_BRANCH="last-working-ci-before-0p16"
PROBLEMATIC_COMMIT="ffc6131"

main() {
    log_info "Creating permanent branch with repaired notebooks"
    echo
    log_info "This will create branch '$BRANCH_NAME' containing:"
    log_info "• Fixed notebooks that work with HARK after breaking changes"
    log_info "• Environment setup demonstrating the problem and solution"
    log_info "• All demonstration scripts for reproducing the issue"
    echo
    
    # Get current directory for reference
    local current_dir=$(pwd)
    log_info "Current problematic directory: $current_dir"
    
    # Navigate back to the main repository (one level up)
    log_info "Navigating to main repository..."
    cd ..
    
    # Ensure we're on the correct base branch
    log_info "Switching to base branch: $BASE_BRANCH"
    git checkout "$BASE_BRANCH"
    
    # Create the new branch
    log_info "Creating new branch: $BRANCH_NAME"
    if git branch | grep -q "$BRANCH_NAME"; then
        log_warning "Branch $BRANCH_NAME already exists. Deleting and recreating..."
        git branch -D "$BRANCH_NAME"
    fi
    
    # Start from the problematic commit to get the right notebook content
    git checkout -b "$BRANCH_NAME" "$PROBLEMATIC_COMMIT"
    
    # Copy the fixed notebooks from the problematic directory
    log_info "Copying fixed notebooks from problematic directory..."
    cp "$current_dir/notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb" notebooks/
    
    # Copy the environment configuration that demonstrates the breaking change
    log_info "Setting up environment configuration..."
    cp "$current_dir/binder/environment.yml" binder/
    
    # Copy the demonstration scripts
    log_info "Adding demonstration scripts..."
    cp "$current_dir/test_problematic_notebooks.sh" .
    cp "$current_dir/fix_problematic_notebooks.sh" .
    if [ -f "$current_dir/.envrc" ]; then
        cp "$current_dir/.envrc" .
    fi
    
    # Create a comprehensive README for this branch
    cat > README_REPAIRED_BRANCH.md << 'EOF'
# Repaired Scripts After Breaking Change

This branch demonstrates the complete solution to the HARK breaking change issue that was masked by CI environment caching.

## Background

- **May 22, 2024**: HARK commit `7a6e8f39` moved `HARK.datasets` to `HARK.Calibration`
- **September 18, 2024**: DemARK commit `ffc6131` appeared successful in CI but had old imports
- **Problem**: Environment caching masked the breaking change for 4 months

## What This Branch Contains

### Fixed Notebooks
- `notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb` - Updated with correct imports
- All other notebooks work without changes

### Environment Setup
- `binder/environment.yml` - Configured with HARK commit `7a6e8f39` (the breaking change)
- `.envrc` - Automatic environment activation with direnv

### Demonstration Scripts
- `test_problematic_notebooks.sh` - Tests the two affected notebooks
- `fix_problematic_notebooks.sh` - Applies and verifies the fixes

## Usage

1. **Setup environment** (first time only):
   ```bash
   mamba env create -f binder/environment.yml
   direnv allow  # If you have direnv installed
   ```

2. **Test the original problem**:
   ```bash
   # First revert to old imports to see the problem
   git show HEAD~1:notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb > notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb
   ./test_problematic_notebooks.sh
   ```

3. **Apply and verify the fix**:
   ```bash
   ./fix_problematic_notebooks.sh
   ```

## The Fix

The breaking change moved HARK modules:
```python
# BEFORE (broken):
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table

# AFTER (fixed):
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
```

## Environment Details

- **HARK Version**: 0.14.1 (commit `7a6e8f39` - the breaking change)
- **Python**: 3.10
- **Environment Name**: `DemARK_problematic_combo`

This environment demonstrates what CI should have been testing if not for caching.
EOF
    
    # Add all changes to git
    log_info "Adding changes to git..."
    git add .
    
    # Commit the changes
    log_info "Committing repaired notebooks and demonstration setup..."
    git commit -m "Add repaired notebooks and demonstration setup

- Fix LC-Model notebook imports: HARK.datasets → HARK.Calibration
- Add environment.yml with HARK commit 7a6e8f39 (breaking change)
- Add demonstration scripts for testing problem and solution
- Add .envrc for automatic environment activation
- Add comprehensive README explaining the issue and fix

This branch demonstrates the complete solution to the CI caching issue
that masked HARK breaking changes for 4 months."
    
    # Return to original directory
    cd "$current_dir"
    
    # Final success message
    echo
    log_success "🎉 SUCCESS: Created branch '$BRANCH_NAME'"
    echo
    log_info "=== WHAT WAS CREATED ==="
    log_info "✅ New git branch: $BRANCH_NAME"
    log_info "✅ Fixed notebooks with correct HARK.Calibration imports"
    log_info "✅ Environment setup that demonstrates the breaking change"
    log_info "✅ Demonstration scripts for testing and fixing"
    log_info "✅ Automatic environment activation with direnv"
    log_info "✅ Comprehensive documentation"
    echo
    log_info "=== HOW TO USE ==="
    log_info "Anyone can now:"
    log_info "1. git checkout $BRANCH_NAME"
    log_info "2. cd into the repository root"
    log_info "3. mamba env create -f binder/environment.yml"
    log_info "4. direnv allow (if direnv is installed)"
    log_info "5. Run ./test_problematic_notebooks.sh and ./fix_problematic_notebooks.sh"
    echo
    log_info "This provides a complete, reproducible demonstration of:"
    log_info "• The CI caching problem that masked breaking changes"
    log_info "• The exact fix needed (HARK.datasets → HARK.Calibration)"
    log_info "• Verification that the fix works"
    echo
    log_warning "Note: You are still in the problematic working directory"
    log_warning "To switch to the new branch: cd .. && git checkout $BRANCH_NAME"
}

main "$@" 
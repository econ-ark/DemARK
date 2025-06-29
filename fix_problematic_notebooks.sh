#!/usr/bin/env bash
# fix_problematic_notebooks.sh
# Purpose: Apply fixes to the notebooks affected by HARK breaking changes and verify they work
#
# This script fixes the import statements that were broken when HARK moved datasets to Calibration

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

apply_fixes() {
    log_info "Applying fixes to problematic notebooks..."
    
    # Fix LC-Model notebook
    log_info "Fixing LC-Model-Expected-Vs-Realized-Income-Growth.ipynb..."
    if grep -q "from HARK.datasets" notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb; then
        sed -i '' 's/from HARK\.datasets\.life_tables\.us_ssa\.SSATools import/from HARK.Calibration.life_tables.us_ssa.SSATools import/g' notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb
        log_success "✅ Fixed HARK.datasets → HARK.Calibration in LC-Model notebook"
    else
        log_info "LC-Model notebook already has correct imports"
    fi
    
    # Check Micro-and-Macro notebook (should not need fixes)
    log_info "Checking Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb..."
    if grep -q "from HARK.datasets" notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb; then
        log_warning "Found unexpected HARK.datasets import in Micro-and-Macro notebook"
        sed -i '' 's/from HARK\.datasets/from HARK.Calibration/g' notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb
        log_success "✅ Fixed HARK.datasets → HARK.Calibration in Micro-and-Macro notebook"
    else
        log_success "✅ Micro-and-Macro notebook has correct imports (no changes needed)"
    fi
}

test_fixes() {
    log_info "Testing the fixed notebooks..."
    
    # Test LC-Model notebook
    log_info "Testing fixed LC-Model-Expected-Vs-Realized-Income-Growth.ipynb..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb -v; then
        log_success "✅ LC-Model notebook now passes"
    else
        log_error "❌ LC-Model notebook still fails after fix"
        return 1
    fi
    
    echo
    
    # Test Micro-and-Macro notebook
    log_info "Testing Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb -v; then
        log_success "✅ Micro-and-Macro notebook passes"
    else
        log_error "❌ Micro-and-Macro notebook failed"
        return 1
    fi
}

verify_fix() {
    log_info "Verifying the import fix works correctly..."
    
    # Test the corrected import
    if python -c "from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table; print('Import works correctly')"; then
        log_success "✅ Corrected import verified: HARK.Calibration.life_tables.us_ssa.SSATools"
    else
        log_error "❌ Corrected import failed"
        return 1
    fi
    
    # Test that old import still fails (to confirm the problem was real)
    if python -c "from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table" 2>/dev/null; then
        log_warning "⚠️  Old import still works - this suggests the breaking change may not be present"
    else
        log_success "✅ Old import correctly fails: HARK.datasets module not found"
    fi
}

main() {
    log_info "Fixing notebooks affected by HARK breaking changes"
    echo
    log_info "The fix: Change 'HARK.datasets' imports to 'HARK.Calibration'"
    log_info "Reason: HARK commit 7a6e8f39 (May 22, 2024) moved datasets to Calibration"
    echo
    
    # Step 1: Apply the fixes
    apply_fixes
    echo
    
    # Step 2: Verify the import fix works
    verify_fix
    echo
    
    # Step 3: Test the fixed notebooks
    test_fixes
    echo
    
    # Final summary
    log_success "🎉 SUCCESS: All fixes applied and verified"
    echo
    log_info "=== SUMMARY ==="
    log_info "✅ Applied import fixes: HARK.datasets → HARK.Calibration"
    log_info "✅ Verified corrected imports work"
    log_info "✅ Confirmed both problematic notebooks now pass"
    echo
    log_info "The notebooks are now compatible with HARK versions that have the breaking changes"
    log_info "This demonstrates how to resolve the CI failures that were masked by environment caching"
}

main "$@" 
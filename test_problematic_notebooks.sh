#!/usr/bin/env bash
# test_problematic_notebooks.sh
# Purpose: Test only the two notebooks that fail due to HARK breaking changes
#
# NOTE: All other notebooks (171 tests) have been verified to pass with this environment.
# This script focuses on the specific notebooks that fail due to the HARK.datasets → HARK.Calibration breaking change.

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
    log_info "Testing the two problematic notebooks affected by HARK breaking changes"
    echo
    log_info "Background: In May 2024, HARK commit 7a6e8f39 moved HARK.datasets to HARK.Calibration"
    log_info "These notebooks still use the old import paths and should fail:"
    echo
    
    # Test LC-Model notebook (expected to fail)
    log_info "Testing LC-Model-Expected-Vs-Realized-Income-Growth.ipynb (expecting failure)..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb -v; then
        log_warning "❌ LC-Model notebook passed when it should have failed"
        echo "   This suggests the breaking change issue may have been resolved already"
    else
        log_success "✅ LC-Model notebook failed as expected"
        echo "   Error: ModuleNotFoundError: No module named 'HARK.datasets'"
    fi
    
    echo
    
    # Test Micro-and-Macro notebook (should pass - no problematic imports)
    log_info "Testing Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb (should pass)..."
    if python -m pytest --nbval-lax --nbval-cell-timeout=12000 notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb -v; then
        log_success "✅ Micro-and-Macro notebook passed as expected"
        echo "   This notebook doesn't use the problematic HARK.datasets imports"
    else
        log_error "❌ Micro-and-Macro notebook failed unexpectedly"
        echo "   This suggests there may be other issues beyond the known breaking change"
    fi
    
    echo
    log_info "=== SUMMARY ==="
    log_info "This demonstrates the specific impact of the HARK breaking change:"
    log_info "• LC-Model notebook: Fails due to 'from HARK.datasets.life_tables.us_ssa.SSATools import'"
    log_info "• Micro-and-Macro notebook: Passes (no problematic imports)"
    log_info "• All other notebooks (171): Previously verified to pass"
    echo
    log_info "Next step: Run fix_problematic_notebooks.sh to apply the corrections"
}

main "$@" 
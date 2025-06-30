#!/bin/bash
# test_ci_locally.sh
# Script to replicate CI tests locally for debugging

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

# Function to test the current environment
test_current_environment() {
    log_info "Testing current environment..."
    
    # Check if conda/mamba is available
    if command -v mamba >/dev/null 2>&1; then
        CONDA_CMD="mamba"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_CMD="conda"
    else
        log_error "Neither conda nor mamba found. Please install conda/miniconda first."
        exit 1
    fi
    
    log_info "Using: $CONDA_CMD"
    
    # Check if DemARK environment exists
    if $CONDA_CMD env list | grep -q "DemARK"; then
        log_success "DemARK environment found"
        
        # Activate environment and test
        log_info "Activating DemARK environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate DemARK
        
        # Test Python and key packages
        log_info "Testing Python and key packages..."
        python -c "
import sys
print(f'Python version: {sys.version}')

# Test core packages
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy
    print('✅ Core scientific packages imported successfully')
except ImportError as e:
    print(f'❌ Core package import failed: {e}')
    sys.exit(1)

# Test HARK
try:
    import HARK
    print(f'✅ HARK imported successfully (version: {HARK.__version__})')
    
    # Test the specific import that was causing issues
    try:
        from HARK.Calibration import load_SCF_wealth_weights
        print('✅ HARK.Calibration import works (fixed version)')
    except ImportError:
        try:
            from HARK.datasets import load_SCF_wealth_weights
            print('⚠️  Using old HARK.datasets import (pre-v0.16)')
        except ImportError as e:
            print(f'❌ Both HARK import methods failed: {e}')
            
except ImportError as e:
    print(f'❌ HARK import failed: {e}')
    sys.exit(1)
"
        
        if [ $? -eq 0 ]; then
            log_success "Environment test passed"
            return 0
        else
            log_error "Environment test failed"
            return 1
        fi
    else
        log_warning "DemARK environment not found. Creating it..."
        return 1
    fi
}

# Function to create environment from current environment.yml
create_environment() {
    log_info "Creating DemARK environment from binder/environment.yml..."
    
    if [ ! -f "binder/environment.yml" ]; then
        log_error "binder/environment.yml not found"
        exit 1
    fi
    
    log_info "Environment file contents:"
    cat binder/environment.yml
    echo
    
    # Remove existing environment if it exists
    $CONDA_CMD env remove -n DemARK --yes 2>/dev/null || true
    
    # Create environment with timeout and retry logic
    log_info "Creating environment (this may take several minutes)..."
    if timeout 600 $CONDA_CMD env create -f binder/environment.yml; then
        log_success "Environment created successfully"
    else
        log_warning "Environment creation timed out or failed. Retrying with simpler approach..."
        
        # Try creating a minimal environment first
        log_info "Creating minimal environment..."
        $CONDA_CMD create -n DemARK python=3.10 -y
        
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate DemARK
        
        # Install packages one by one
        log_info "Installing packages individually..."
        $CONDA_CMD install -n DemARK -c conda-forge matplotlib numpy ipywidgets seaborn scipy pandas statsmodels tqdm pytest nbval -y
        
        # Install HARK via pip
        log_info "Installing HARK via pip..."
        pip install git+https://github.com/econ-ark/hark@master
        
        log_success "Minimal environment created"
    fi
}

# Function to test notebooks
test_notebooks() {
    log_info "Testing notebooks..."
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate DemARK
    
    # Test the specific notebooks that were having issues
    local test_notebooks=(
        "notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb"
        "notebooks/Gentle-Intro-To-HARK-PerfForesightCRRA.ipynb"
    )
    
    for notebook in "${test_notebooks[@]}"; do
        if [ -f "$notebook" ]; then
            log_info "Testing $(basename "$notebook")..."
            if python -m pytest --nbval-lax --nbval-cell-timeout=60 "$notebook" -v; then
                log_success "✅ $(basename "$notebook") passed"
            else
                log_error "❌ $(basename "$notebook") failed"
            fi
        else
            log_warning "Notebook not found: $notebook"
        fi
    done
}

# Function to test the CI build workflow
test_ci_build() {
    log_info "Testing CI build workflow..."
    
    # Check if we can replicate the CI build steps
    if [ -f ".github/workflows/build.yml" ]; then
        log_info "Found CI workflow file"
        
        # Extract the test command from the workflow
        log_info "CI test command:"
        grep -A 10 "Test with nbval" .github/workflows/build.yml || log_warning "Could not find test command in workflow"
        
        # Run the equivalent test
        log_info "Running CI-equivalent test..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate DemARK
        
        python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
          --ignore=notebooks/Chinese-Growth.ipynb \
          --ignore=notebooks/Harmenberg-Aggregation.ipynb \
          notebooks/ -v
    else
        log_warning "CI workflow file not found"
    fi
}

# Function to diagnose devcontainer issues
diagnose_devcontainer() {
    log_info "Diagnosing devcontainer issues..."
    
    # Check Docker
    if command -v docker >/dev/null 2>&1; then
        log_success "Docker is available: $(docker --version)"
        
        # Check if we can run a simple container
        log_info "Testing basic Docker functionality..."
        if docker run --rm hello-world >/dev/null 2>&1; then
            log_success "Docker is working correctly"
        else
            log_warning "Docker basic test failed"
        fi
    else
        log_error "Docker not found"
    fi
    
    # Check devcontainer files
    if [ -d ".devcontainer" ]; then
        log_success "Devcontainer directory found"
        log_info "Devcontainer files:"
        ls -la .devcontainer/
        
        # Validate devcontainer.json
        if [ -f ".devcontainer/devcontainer.json" ]; then
            log_info "Validating devcontainer.json..."
            if python -m json.tool .devcontainer/devcontainer.json >/dev/null 2>&1; then
                log_success "devcontainer.json is valid JSON"
            else
                log_error "devcontainer.json has syntax errors"
            fi
        fi
        
        # Check Dockerfile
        if [ -f ".devcontainer/Dockerfile" ]; then
            log_info "Dockerfile found"
            log_info "Dockerfile base image:"
            head -5 .devcontainer/Dockerfile
        fi
    else
        log_error "Devcontainer directory not found"
    fi
}

# Main function
main() {
    echo "🔍 DemARK CI Local Testing Script"
    echo "================================="
    echo
    
    log_info "Current directory: $(pwd)"
    log_info "Git branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
    echo
    
    # Test current environment first
    if test_current_environment; then
        log_success "Current environment is working"
    else
        log_info "Creating new environment..."
        create_environment
        
        if test_current_environment; then
            log_success "New environment is working"
        else
            log_error "Environment setup failed"
            exit 1
        fi
    fi
    
    echo
    log_info "=== TESTING NOTEBOOKS ==="
    test_notebooks
    
    echo
    log_info "=== TESTING CI BUILD ==="
    test_ci_build
    
    echo
    log_info "=== DIAGNOSING DEVCONTAINER ==="
    diagnose_devcontainer
    
    echo
    log_success "🎉 Local CI testing complete!"
    log_info "This helps replicate and debug CI issues without requiring full devcontainer builds"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo "Test DemARK CI locally to debug failures"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac 
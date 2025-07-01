#!/bin/bash
# test_ci_locally.sh
# Script to replicate CI testing locally for DemARK notebooks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
ENV_NAME="DemARK_CI_test"
TIMEOUT="12000"

log_info "🧪 Starting local CI test replication..."

# Check if environment exists, create if not
if ! conda env list | grep -q "^${ENV_NAME} "; then
    log_info "Creating fresh CI test environment: ${ENV_NAME}"
    mamba env create -f binder/environment.yml -n "${ENV_NAME}"
else
    log_info "Using existing environment: ${ENV_NAME}"
fi

# Activate environment
log_info "Activating environment: ${ENV_NAME}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Show HARK version
log_info "Checking HARK version..."
python -c "import HARK; print(f'HARK version: {HARK.__version__}')"

# Run the exact CI command
log_info "🚀 Running CI test command..."
log_info "Command: python -m pytest --nbval-lax --nbval-cell-timeout=${TIMEOUT} --ignore=notebooks/Chinese-Growth.ipynb --ignore=notebooks/Harmenberg-Aggregation.ipynb notebooks/ -v"

start_time=$(date +%s)

# Run the test
if python -m pytest --nbval-lax --nbval-cell-timeout="${TIMEOUT}" \
    --ignore=notebooks/Chinese-Growth.ipynb \
    --ignore=notebooks/Harmenberg-Aggregation.ipynb \
    notebooks/ -v; then
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    log_success "✅ All tests passed in ${minutes}m ${seconds}s"
    exit 0
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    log_error "❌ Tests failed after ${minutes}m ${seconds}s"
    exit 1
fi 
#!/usr/bin/env bash
# bisect_hark_breaking_changes.sh
# Purpose: Use git bisection to find the exact HARK commit that breaks DemARK notebooks
#
# This script automates the process of finding breaking changes in HARK that affect DemARK

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
HARK_REPO_PATH="../../econ-ark/HARK"  # Adjust path to your HARK repository
DEMARK_REPO_PATH="."         # Current DemARK repository
ENVIRONMENT_NAME="DemARK_bisect_test"  # Temporary environment for testing
GOOD_COMMIT=""               # Will be set by user input or defaults
BAD_COMMIT=""                # Will be set by user input or defaults
TEST_NOTEBOOKS=(
    "notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb"
    "notebooks/Micro-and-Macro-Implications-of-Very-Impatient-HHs.ipynb"
    # Add more notebooks to test here
)

# Default commits - these can be overridden
DEFAULT_GOOD_COMMIT="7a6e8f39"  # May 22, 2024 - HARK commit that moved datasets to Calibration (we fixed this)
DEFAULT_BAD_COMMIT="HEAD"       # Current HARK master - may have new breaking changes

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Perform git bisection to find NEW HARK commits that break DemARK notebooks.
This version starts from the resolved datasets→Calibration breaking change (7a6e8f39)
and bisects forward to find any additional breaking changes in recent HARK versions.

OPTIONS:
    -g, --good COMMIT     Known good HARK commit (default: $DEFAULT_GOOD_COMMIT)
    -b, --bad COMMIT      Known bad HARK commit (default: $DEFAULT_BAD_COMMIT)
    -h, --help           Show this help message
    --hark-path PATH     Path to HARK repository (default: $HARK_REPO_PATH)
    --notebooks LIST     Comma-separated list of notebooks to test
    --dry-run           Show what would be tested without running bisection

EXAMPLES:
    # Basic bisection from 7a6e8f39 (resolved breaking change) to HEAD (current)
    $0

    # Specify different commit range
    $0 --good 7a6e8f39 --bad HEAD

    # Test specific notebooks only
    $0 --notebooks "notebooks/LC-Model-Expected-Vs-Realized-Income-Growth.ipynb"

    # Dry run to see what would be tested
    $0 --dry-run

PREREQUISITES:
    1. HARK repository must be available at specified path
    2. Must be on repaired-scripts-after-breaking-change branch with fixed imports
    3. Test notebooks must exist and have resolved the datasets→Calibration issue
    4. mamba/conda must be available for environment creation

The script will:
    1. Validate the setup
    2. Start git bisection in the HARK repository (7a6e8f39 → HEAD)
    3. For each commit, create a fresh environment with that HARK version
    4. Test current DemARK notebooks with fixed import paths
    5. Automatically mark commits as good/bad based on test results
    6. Report any new breaking changes introduced after the resolved datasets issue
    7. Clean up temporary environments

APPROACH:
    - Starts from HARK commit 7a6e8f39 (May 2024, datasets→Calibration change we've fixed)
    - Tests forward to current HARK HEAD (latest master, may have new breaking changes)
    - Creates isolated conda environments for each HARK commit
    - Tests current DemARK notebooks with fixed imports to find new breaking changes
EOF
}

validate_setup() {
    log_info "Validating bisection setup..."
    
    # Check HARK repository exists
    if [ ! -d "$HARK_REPO_PATH" ]; then
        log_error "HARK repository not found at: $HARK_REPO_PATH"
        log_info "Please clone HARK or adjust --hark-path"
        return 1
    fi
    
    # Check HARK repository is a git repo
    if [ ! -d "$HARK_REPO_PATH/.git" ]; then
        log_error "HARK path is not a git repository: $HARK_REPO_PATH"
        return 1
    fi
    
    # Check test notebooks exist
    for notebook in "${TEST_NOTEBOOKS[@]}"; do
        if [ ! -f "$notebook" ]; then
            log_error "Test notebook not found: $notebook"
            return 1
        fi
    done
    
    # Check conda environment
    if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
        log_error "No conda environment active"
        log_info "Please activate a DemARK environment first"
        return 1
    fi
    
    log_success "Setup validation passed"
    return 0
}

create_test_environment() {
    local commit_short=$1
    local env_name="${ENVIRONMENT_NAME}_${commit_short}"
    
    # Use plain echo to avoid color codes in output
    echo "Creating test environment: $env_name" >&2
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^$env_name "; then
        echo "Removing existing environment: $env_name" >&2
        conda env remove -n "$env_name" -y --quiet >/dev/null 2>&1
    fi
    
    # Create new environment with current HARK commit
    echo "Creating environment with HARK commit $commit_short..." >&2
    
    # Create a temporary environment.yml with the specific HARK commit
    cat > /tmp/bisect_env_${commit_short}.yml << EOF
name: ${env_name}
channels:
  - conda-forge
  - pyviz
dependencies:
  - python=3.10
  - pip
  - jupyter
  - matplotlib
  - numpy
  - scipy
  - pandas
  - numba
  - pytest
  - nbval
  - pip:
    - git+https://github.com/econ-ark/HARK.git@${commit_short}
    - econ-ark
    - quantecon
    - statsmodels
    - linearmodels
EOF
    
    # Create the environment (redirect all output to stderr)
    if ! mamba env create -f /tmp/bisect_env_${commit_short}.yml >&2; then
        echo "Failed to create environment for HARK commit $commit_short" >&2
        rm -f /tmp/bisect_env_${commit_short}.yml
        return 1
    fi
    
    # Clean up temp file
    rm -f /tmp/bisect_env_${commit_short}.yml
    
    # Return just the environment name to stdout (no other output)
    echo "$env_name"
    return 0
}

test_current_hark_commit() {
    local commit_hash=$(cd "$HARK_REPO_PATH" && git rev-parse HEAD 2>/dev/null)
    local commit_short=$(cd "$HARK_REPO_PATH" && git rev-parse --short HEAD 2>/dev/null)
    local commit_msg=$(cd "$HARK_REPO_PATH" && git log -1 --pretty=format:"%s" 2>/dev/null)
    
    if [ -z "$commit_short" ]; then
        log_error "Failed to get current HARK commit information"
        return 1
    fi
    
    log_info "Testing HARK commit: $commit_short - $commit_msg"
    
    # Create test environment with this HARK commit
    local test_env
    if ! test_env=$(create_test_environment "$commit_short"); then
        log_error "Failed to create test environment for HARK commit $commit_short"
        log_warning "This might indicate the commit doesn't exist or has build issues"
        return 1
    fi
    
    log_info "Using test environment: $test_env"
    
    # Test each notebook in the new environment
    # Use current notebooks with fixed import paths
    local failed_notebooks=()
    for notebook in "${TEST_NOTEBOOKS[@]}"; do
        log_info "Testing notebook: $(basename "$notebook")"
        
        # Use current version of notebook with fixed imports
        local abs_notebook="$DEMARK_REPO_PATH/$notebook"
        if conda run -n "$test_env" --cwd "$DEMARK_REPO_PATH" python -m pytest --nbval-lax --nbval-cell-timeout=12000 "$abs_notebook" -v --tb=no -q; then
            log_success "✅ $(basename "$notebook") passed"
        else
            log_error "❌ $(basename "$notebook") failed"
            failed_notebooks+=("$notebook")
        fi
    done
    
    # Clean up test environment
    log_info "Cleaning up test environment: $test_env"
    conda env remove -n "$test_env" -y --quiet
    
    # Return status
    if [ ${#failed_notebooks[@]} -eq 0 ]; then
        log_success "All notebooks passed with HARK commit $commit_short"
        return 0  # Good commit
    else
        log_error "Failed notebooks with HARK commit $commit_short:"
        for notebook in "${failed_notebooks[@]}"; do
            log_error "  - $(basename "$notebook")"
        done
        return 1  # Bad commit
    fi
}

run_bisection() {
    log_info "Starting git bisection in HARK repository"
    log_info "Good commit: $GOOD_COMMIT"
    log_info "Bad commit: $BAD_COMMIT"
    log_info "Testing notebooks: ${TEST_NOTEBOOKS[*]}"
    echo
    
    # Navigate to HARK repository
    cd "$HARK_REPO_PATH"
    
    # Start bisection
    log_info "Initializing git bisect..."
    git bisect start
    
    # Mark the commits
    log_info "Marking bad commit: $BAD_COMMIT"
    git bisect bad "$BAD_COMMIT"
    
    log_info "Marking good commit: $GOOD_COMMIT"
    git bisect good "$GOOD_COMMIT"
    
    # Return to DemARK directory for testing
    cd "$DEMARK_REPO_PATH"
    
    # Run the bisection
    log_info "Running automated bisection..."
    echo
    
    # The bisection loop with safety counter
    local iteration=0
    local max_iterations=20  # Safety limit
    while true; do
        iteration=$((iteration + 1))
        if [ $iteration -gt $max_iterations ]; then
            log_error "Bisection exceeded maximum iterations ($max_iterations). Stopping."
            break
        fi
        
        log_info "Bisection iteration $iteration"
        # Test current commit
        if test_current_hark_commit; then
            # Good commit
            cd "$HARK_REPO_PATH"
            local bisect_output
            bisect_output=$(git bisect good 2>&1)
            local result=$?
            echo "$bisect_output"
        else
            # Bad commit
            cd "$HARK_REPO_PATH"
            local bisect_output
            bisect_output=$(git bisect bad 2>&1)
            local result=$?
            echo "$bisect_output"
        fi
        
        cd "$DEMARK_REPO_PATH"
        
        # Check if bisection is complete (git bisect outputs specific messages)
        if echo "$bisect_output" | grep -q "is the first bad commit\|There are only 'skip'ped commits left\|We cannot bisect more"; then
            log_info "Bisection complete!"
            break
        fi
        
        # Check for bisection errors
        if [ $result -ne 0 ] && ! echo "$bisect_output" | grep -q "Bisecting:"; then
            log_error "Git bisect failed: $bisect_output"
            break
        fi
        
        echo
        log_info "Continuing bisection..."
    done
    
    # Get the final result
    cd "$HARK_REPO_PATH"
    local breaking_commit=$(git bisect view --pretty=format:"%H")
    local breaking_commit_short=$(git rev-parse --short "$breaking_commit")
    local breaking_commit_msg=$(git log -1 --pretty=format:"%s" "$breaking_commit")
    local breaking_commit_date=$(git log -1 --pretty=format:"%ad" --date=short "$breaking_commit")
    
    # Clean up bisection
    git bisect reset
    
    cd "$DEMARK_REPO_PATH"
    
    # Report results
    echo
    log_success "🎯 BISECTION COMPLETE"
    echo
    log_info "=== BREAKING CHANGE FOUND ==="
    log_info "Commit: $breaking_commit"
    log_info "Short: $breaking_commit_short"
    log_info "Date: $breaking_commit_date"
    log_info "Message: $breaking_commit_msg"
    echo
    log_info "This commit introduced breaking changes that cause these notebooks to fail:"
    for notebook in "${TEST_NOTEBOOKS[@]}"; do
        log_info "  - $(basename "$notebook")"
    done
    echo
    log_info "You can examine this commit with:"
    log_info "  cd $HARK_REPO_PATH && git show $breaking_commit_short"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--good)
            GOOD_COMMIT="$2"
            shift 2
            ;;
        -b|--bad)
            BAD_COMMIT="$2"
            shift 2
            ;;
        --hark-path)
            HARK_REPO_PATH="$2"
            shift 2
            ;;
        --notebooks)
            IFS=',' read -ra TEST_NOTEBOOKS <<< "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set defaults if not provided
GOOD_COMMIT=${GOOD_COMMIT:-$DEFAULT_GOOD_COMMIT}
BAD_COMMIT=${BAD_COMMIT:-$DEFAULT_BAD_COMMIT}

main() {
    log_info "HARK Breaking Change Bisection Tool"
    echo
    
    # Validate setup
    if ! validate_setup; then
        exit 1
    fi
    
    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_info "DRY RUN - Would test:"
        log_info "  HARK repository: $HARK_REPO_PATH"
        log_info "  Good commit: $GOOD_COMMIT"
        log_info "  Bad commit: $BAD_COMMIT"
        log_info "  Test notebooks:"
        for notebook in "${TEST_NOTEBOOKS[@]}"; do
            log_info "    - $notebook"
        done
        exit 0
    fi
    
    # Run the bisection
    run_bisection
}

main "$@" 
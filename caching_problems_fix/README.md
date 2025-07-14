# DemARK CI Caching Problems Fix

This directory contains diagnostic and testing tools developed to investigate and resolve the DemARK CI caching issues that masked HARK v0.16+ compatibility problems for 11 months.

## 🎯 Problem Summary

**Root Cause**: GitHub Actions cached a conda environment from November 2023 that contained HARK v0.13.0. When HARK v0.16.0 introduced breaking changes (moving `HARK.datasets` → `HARK.Calibration`), CI continued passing because it used the old cached environment instead of creating fresh environments with current HARK versions.

**Impact**: DemARK notebooks were broken for 11 months for anyone doing fresh installations, while CI falsely reported success.

## 📁 Script Overview

### Core Diagnostic Scripts

#### `bisect_hark_breaking_changes.sh`
**Purpose**: Automated git bisection to identify the exact HARK commit that broke DemARK compatibility.

**What it does**:
- Performs automated git bisection on HARK repository
- Tests each HARK commit against DemARK notebooks
- Identifies commit `7a6e8f39` (May 22, 2024) as the breaking change
- Documents the `datasets` → `Calibration` module reorganization

**Usage**:
```bash
./bisect_hark_breaking_changes.sh
./bisect_hark_breaking_changes.sh --help  # Show options
```

**Relates to**: This script provided the forensic evidence needed to understand exactly when and why the compatibility broke.

#### `last-working-ci-before-0p16-preserve.sh`
**Purpose**: Creates side-by-side environments to reproduce both the "working" cached state and the current fixed state.

**What it does**:
- Creates `DemARK_historical_working` environment (HARK v0.13.0, old imports)
- Creates `DemARK_current_fixed` environment (HARK master, new imports)
- Sets up git worktrees for testing different DemARK versions
- Provides automated testing scripts with environment activation
- Includes direnv integration for automatic environment switching

**Usage**:
```bash
./last-working-ci-before-0p16-preserve.sh
# Creates both environments and provides testing instructions
```

**Relates to**: This is the main reproduction script that demonstrates the caching issue and validates the fix.

#### `reproduce-problematic-combo.sh`
**Purpose**: Demonstrates the specific combination that was falsely appearing successful due to caching.

**What it does**:
- Creates environment with HARK commit `7a6e8f39` (first breaking change)
- Tests old DemARK imports against new HARK (shows failures)
- Repairs the imports (`HARK.datasets` → `HARK.Calibration`)
- Validates that repaired version works correctly

**Usage**:
```bash
./reproduce-problematic-combo.sh
# Shows the failure, applies fix, demonstrates success
```

**Relates to**: This script proves that the CI was only successful due to caching, not because the code actually worked.

### Testing and Validation Scripts

#### `test_ci_locally.sh`
**Purpose**: Replicates CI testing environment locally for debugging and validation.

**What it does**:
- Creates fresh DemARK environment from `binder/environment.yml`
- Tests HARK import compatibility (both old and new paths)
- Runs the exact CI test command locally
- Validates notebook execution with current HARK versions
- Provides detailed diagnostic output

**Usage**:
```bash
./test_ci_locally.sh
# Runs comprehensive local CI replication
```

**Relates to**: This script ensures that our fixes actually work in a CI-equivalent environment.

#### `test_complete_setup.sh`
**Purpose**: Automated end-to-end setup and testing script for complete workflow validation.

**What it does**:
- Handles conda initialization and environment cleanup
- Runs the main environment creation script
- Validates that all environments were created correctly
- Sets up direnv for automatic environment switching
- Provides step-by-step verification instructions

**Usage**:
```bash
./test_complete_setup.sh
# Fully automated setup and validation
```

**Relates to**: This is the "one-click" solution for reproducing and validating the entire caching issue investigation.

### Setup and Configuration Scripts

#### `setup_conda.sh`
**Purpose**: Handles conda initialization and configuration for all other scripts.

**What it does**:
- Detects current shell (bash/zsh)
- Initializes conda for the detected shell
- Sets up direnv integration if available
- Provides setup instructions for manual configuration

**Usage**:
```bash
./setup_conda.sh
# Run once to configure conda for other scripts
```

**Relates to**: This is a prerequisite script that ensures conda is properly configured before running other diagnostic tools.

## 🔄 Workflow and Script Relationships

### Investigation Workflow
1. **`setup_conda.sh`** - Initial conda configuration
2. **`bisect_hark_breaking_changes.sh`** - Identify the breaking commit
3. **`reproduce-problematic-combo.sh`** - Demonstrate the masked failure
4. **`last-working-ci-before-0p16-preserve.sh`** - Create reproduction environments
5. **`test_ci_locally.sh`** - Validate fixes work in CI-equivalent environment

### Validation Workflow
1. **`test_complete_setup.sh`** - Automated end-to-end setup
2. **`test_ci_locally.sh`** - Verify CI compatibility
3. Manual testing in created environments

### Script Dependencies
```
setup_conda.sh
    ↓ (conda configuration)
bisect_hark_breaking_changes.sh
    ↓ (identifies breaking commit)
reproduce-problematic-combo.sh
    ↓ (demonstrates issue)
last-working-ci-before-0p16-preserve.sh
    ↓ (creates test environments)
test_ci_locally.sh
    ↓ (validates fixes)
test_complete_setup.sh (orchestrates entire workflow)
```

## 🎯 Goals and Achievements

### Primary Goals
1. **Identify Root Cause**: ✅ Found 11-month CI caching issue masking HARK compatibility
2. **Reproduce Issue**: ✅ Created environments that demonstrate both failure and success states
3. **Fix Compatibility**: ✅ Updated all DemARK imports for HARK v0.16+
4. **Prevent Recurrence**: ✅ Disabled CI environment caching (`cache-environment: false`)
5. **Provide Diagnostics**: ✅ Created comprehensive toolset for future debugging

### Secondary Goals
1. **Documentation**: ✅ Comprehensive README and script documentation
2. **Automation**: ✅ One-click reproduction and testing scripts
3. **Developer Experience**: ✅ direnv integration for automatic environment switching
4. **CI Integration**: ✅ Updated workflows and devcontainer configuration

## 🚀 Quick Start

### For Investigators (Understanding the Problem)
```bash
# 1. Set up conda
./setup_conda.sh
# Restart shell, then:

# 2. See the forensic evidence
./bisect_hark_breaking_changes.sh

# 3. Reproduce the masked failure
./reproduce-problematic-combo.sh
```

### For Validators (Testing the Solution)
```bash
# 1. Complete automated setup
./test_complete_setup.sh
# Restart shell, then follow the provided verification steps

# 2. Or run individual validation
./test_ci_locally.sh
```

### For Developers (Using the Environments)
```bash
# Create both historical and current environments
./last-working-ci-before-0p16-preserve.sh

# Use the created environments
cd ../DemARK_historical_*    # Test old version
cd ../DemARK_current_*       # Test fixed version
# (Environments auto-activate with direnv)
```

## 🔍 Key Findings

1. **Caching Duration**: 11 months (Nov 2023 - Oct 2024)
2. **Breaking Commit**: HARK `7a6e8f39` (May 22, 2024) - moved `datasets` to `Calibration`
3. **Affected Notebooks**: 8 notebooks with `HARK.distribution` imports
4. **CI Behavior**: False positives due to stale cached environment
5. **User Impact**: Fresh installations failed while CI reported success

## 🛠️ Technical Details

### Environment Specifications
- **Historical**: HARK v0.13.0, Python 3.10, old import paths
- **Current**: HARK master, Python 3.10, new import paths
- **Problematic**: HARK v0.16.0+, old import paths (demonstrates failure)

### Testing Strategy
- **Notebook Validation**: `--nbval-lax` with 12000s timeout
- **Import Testing**: Both `HARK.datasets` and `HARK.Calibration` paths
- **Environment Isolation**: Separate conda environments for each test case
- **Automation**: Scripts handle environment activation and cleanup

### Integration Points
- **CI Workflows**: Updated `.github/workflows/build.yml`
- **Devcontainer**: Full development environment configuration
- **Git Worktrees**: Isolated testing of different DemARK versions
- **direnv**: Automatic environment switching for developer convenience

## 📚 Related Documentation

- **Main README**: `../README.md` - Project overview
- **Devcontainer**: `../.devcontainer/README.md` - Development environment
- **CI Workflows**: `../.github/workflows/` - Automated testing
- **Environment**: `../binder/environment.yml` - Package specifications

## 🧹 Cleanup

When investigation is complete, clean up with:
```bash
# Remove worktrees
git worktree remove ../DemARK_historical_*
git worktree remove ../DemARK_current_*
git worktree remove ../DemARK_problematic_*

# Remove environments
conda env remove -n DemARK_historical_working
conda env remove -n DemARK_current_fixed
conda env remove -n DemARK_problematic_combo
```

---

**Last Updated**: January 2025  
**Status**: Investigation Complete, Solutions Implemented  
**Maintainer**: DemARK Development Team 
#!/bin/bash

# Test script to validate DemARK devcontainer setup
echo "🧪 Testing DemARK Development Container Setup"
echo "============================================="

# Test 1: Environment activation
echo "📦 Testing conda environment activation..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate DemARK
if [ $? -eq 0 ]; then
    echo "✅ Conda environment activated successfully"
else
    echo "❌ Failed to activate conda environment"
    exit 1
fi

# Test 2: Python and key packages
echo "🐍 Testing Python and key packages..."
python -c "
import sys
print(f'Python version: {sys.version}')

# Test core scientific packages
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy
    print('✅ Core scientific packages (numpy, matplotlib, pandas, scipy) imported successfully')
except ImportError as e:
    print(f'❌ Core package import failed: {e}')
    sys.exit(1)

# Test Jupyter
try:
    import jupyter
    import jupyterlab
    print('✅ Jupyter packages imported successfully')
except ImportError as e:
    print(f'⚠️  Jupyter import issue: {e}')

# Test HARK (might fail in some scenarios, which is expected)
try:
    import HARK
    print(f'✅ HARK imported successfully (version: {HARK.__version__})')
except ImportError as e:
    print(f'⚠️  HARK import issue: {e}')
    print('   This might be expected when testing caching issues')
"

if [ $? -ne 0 ]; then
    echo "❌ Python package test failed"
    exit 1
fi

# Test 3: Jupyter availability
echo "📓 Testing Jupyter availability..."
if command -v jupyter >/dev/null 2>&1; then
    echo "✅ Jupyter command available"
    jupyter --version | head -5
else
    echo "❌ Jupyter command not found"
    exit 1
fi

# Test 4: Git configuration
echo "🔧 Testing Git setup..."
if command -v git >/dev/null 2>&1; then
    echo "✅ Git available"
    git --version
    
    # Check if git is configured (optional)
    if git config --global user.name >/dev/null 2>&1; then
        echo "✅ Git user configured: $(git config --global user.name)"
    else
        echo "⚠️  Git user not configured (this is fine for testing)"
    fi
else
    echo "❌ Git not available"
    exit 1
fi

# Test 5: Development tools
echo "🛠️  Testing development tools..."
tools=("black" "ruff" "pytest")
for tool in "${tools[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "✅ $tool available"
    else
        echo "⚠️  $tool not found"
    fi
done

# Test 6: File permissions and workspace
echo "📁 Testing workspace and permissions..."
if [ -w "/workspace" ]; then
    echo "✅ Workspace is writable"
else
    echo "❌ Workspace is not writable"
    exit 1
fi

# Test 7: Diagnostic scripts
echo "🔍 Testing diagnostic scripts..."
if [ -f "/workspace/bisect_hark_breaking_changes.sh" ]; then
    if [ -x "/workspace/bisect_hark_breaking_changes.sh" ]; then
        echo "✅ Bisection script found and executable"
    else
        echo "⚠️  Bisection script found but not executable"
    fi
else
    echo "⚠️  Bisection script not found (might be expected)"
fi

if [ -d "/workspace/caching_problems_fix" ]; then
    echo "✅ Caching analysis directory found"
else
    echo "⚠️  Caching analysis directory not found (might be expected)"
fi

# Test 8: Port availability
echo "🌐 Testing port availability..."
if netstat -ln 2>/dev/null | grep -q ":8888"; then
    echo "⚠️  Port 8888 already in use"
else
    echo "✅ Port 8888 available for Jupyter"
fi

echo ""
echo "🎉 Container validation complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "  2. Test notebooks: python -m pytest --nbval-lax notebooks/"
echo "  3. Run diagnostics: ./bisect_hark_breaking_changes.sh"
echo "" 
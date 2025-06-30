#!/bin/bash

# Post-create script for DemARK devcontainer
echo "🚀 Setting up DemARK development environment..."

# Ensure we're in the right directory
cd /workspace

# Initialize micromamba and activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate DemARK

# Verify environment setup
echo "📦 Verifying conda environment..."
python --version
pip list | grep -E "(numpy|matplotlib|HARK)" || echo "⚠️  Some packages may not be installed"

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
fi

# Set up git configuration (if not already set)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    echo "Please configure git with your details:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'your.email@example.com'"
fi

# Make diagnostic scripts executable
echo "🔧 Making diagnostic scripts executable..."
find caching_problems_fix/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || echo "caching_problems_fix scripts not found"

# Test notebook execution capability
echo "📓 Testing notebook execution capability..."
if command -v jupyter >/dev/null 2>&1; then
    echo "✅ Jupyter is available"
    # Test if we can import key packages
    python -c "
import sys
try:
    import numpy as np
    import matplotlib.pyplot as plt
    print('✅ NumPy and Matplotlib imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')

try:
    import HARK
    print(f'✅ HARK imported successfully (version: {HARK.__version__})')
except ImportError as e:
    print(f'⚠️  HARK import issue: {e}')
    print('   This might be expected if testing caching issues')
"
else
    echo "❌ Jupyter not found"
fi

# Show useful information
echo ""
echo "🎉 DemARK development environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "  - Run notebooks: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "  - Test notebooks: python -m pytest --nbval-lax notebooks/"
echo "  - Test CI locally: caching_problems_fix/test_ci_locally.sh"
echo "  - Bisect HARK issues: caching_problems_fix/bisect_hark_breaking_changes.sh"
echo "  - Analyze caching: cd caching_problems_fix && ./reproduce-problematic-combo.sh"
echo ""
echo "🔍 Useful directories:"
echo "  - notebooks/     - Jupyter notebooks"
echo "  - DemARK_*/      - Historical versions for testing"
echo "    * DemARK_20250628-2309_current   - Current working version"
echo "    * DemARK_20231129-1727_history   - Historical reference version"
echo "    * DemARK_20240918-0003_counter   - Counterfactual test version"
echo "  - caching_problems_fix/ - Diagnostic tools"
echo "" 
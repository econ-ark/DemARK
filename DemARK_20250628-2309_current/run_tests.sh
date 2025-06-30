#!/bin/bash
# Auto-activate environment and run tests
set -euo pipefail

echo "🔧 Activating DemARK_current_fixed environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DemARK_current_fixed

echo "🧪 Running tests in current fixed environment..."
python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
  --ignore=notebooks/Chinese-Growth.ipynb \
  --ignore=notebooks/Harmenberg-Aggregation.ipynb \
  notebooks/ -v

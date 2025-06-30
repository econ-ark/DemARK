#!/bin/bash
# Auto-activate environment and run tests
set -euo pipefail

echo "🔧 Activating DemARK_historical_working environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate DemARK_historical_working

echo "🧪 Running tests in historical environment..."
python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
  --ignore=notebooks/Chinese-Growth.ipynb \
  --ignore=notebooks/Harmenberg-Aggregation.ipynb \
  notebooks/ -v

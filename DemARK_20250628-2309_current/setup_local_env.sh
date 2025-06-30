#!/bin/bash
# Install conda environment locally in this worktree
set -euo pipefail

ENV_DIR="./local_env"
ENV_NAME="DemARK_current_fixed"

echo "🔧 Creating local conda environment in $ENV_DIR..."

# Create environment in local directory
mamba env create -f binder/environment.yml -p "$ENV_DIR"

echo "✅ Local environment created!"
echo "📋 To use:"
echo "   conda activate $ENV_DIR"
echo "   python -m pytest --nbval-lax notebooks/ -v"

# Create activation script
cat > activate_local.sh << 'INNER_EOF'
#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./local_env
echo "🔧 Activated local environment: $(conda info --name)"
INNER_EOF

chmod +x activate_local.sh
echo "💡 Or run: ./activate_local.sh"

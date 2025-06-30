#!/bin/bash
# Install conda environment locally in this worktree
set -euo pipefail

ENV_DIR="./conda_env"

echo "🔧 Creating local conda environment in $ENV_DIR..."

# Create environment in local directory (not global conda envs)
mamba env create -f binder/environment.yml -p "$ENV_DIR"

echo "✅ Local environment created in $ENV_DIR"

# Create auto-activation script
cat > activate_local.sh << 'INNER_EOF'
#!/bin/bash
# Auto-activate the local environment
export CONDA_ENV_PATH="./conda_env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
echo "🔧 Activated local environment: $(basename $PWD)/conda_env"
INNER_EOF

chmod +x activate_local.sh

# Update .envrc to use local environment
echo 'conda activate ./conda_env' > .envrc

echo "💡 Usage:"
echo "   ./activate_local.sh        # Activate local environment"
echo "   conda activate ./conda_env # Direct activation"

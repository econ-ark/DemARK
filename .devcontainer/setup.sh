#!/bin/bash

# DemARK Development Environment Setup Script
# This script helps debug and setup the conda environment with better space management

echo "=== DemARK Development Environment Setup ==="
echo "Current directory: $(pwd)"
echo "User: $(whoami)"

# Check available space
echo "=== Disk Space Check ==="
df -h /tmp
df -h .

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda command not found!"
    echo "Trying to source conda..."
    source /opt/conda/etc/profile.d/conda.sh
fi

echo "Conda version: $(conda --version)"

# Check if environment.yml exists
if [ ! -f "binder/environment.yml" ]; then
    echo "ERROR: binder/environment.yml not found!"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

echo "Found binder/environment.yml"

# Clean conda cache to free up space
echo "Cleaning conda cache..."
conda clean --all -y

# Check if environment already exists
if conda env list | grep -q "DemARK"; then
    echo "DemARK environment already exists, updating..."
    conda env update -f binder/environment.yml --prune
else
    echo "Creating DemARK environment..."
    # Create environment with explicit error handling
    if ! conda env create -f binder/environment.yml; then
        echo "ERROR: Failed to create conda environment"
        echo "Trying to create a minimal environment first..."
        
        # Create minimal environment
        conda create -n DemARK python=3.10 -y
        conda activate DemARK
        
        # Install packages one by one
        echo "Installing packages individually..."
        conda install -c conda-forge matplotlib numpy ipywidgets seaborn scipy pandas jupyter jupyterlab notebook pip -y
        
        # Try to install HARK
        echo "Installing HARK..."
        pip install --no-cache-dir git+https://github.com/econ-ark/hark@v0.16.0
        
        if [ $? -ne 0 ]; then
            echo "WARNING: HARK installation failed, but continuing..."
        fi
    fi
fi

# Initialize conda for bash
echo "Initializing conda..."
conda init bash

# Activate environment
echo "Activating DemARK environment..."
source ~/.bashrc
conda activate DemARK

# Clean up after installation
echo "Cleaning up..."
conda clean --all -y
pip cache purge 2>/dev/null || true

# Verify installation
echo "=== Environment Verification ==="
conda info --envs
echo "Python version: $(python --version)"
echo "Jupyter Lab: $(which jupyter-lab)"

# Check if HARK is available
python -c "import HARK; print('HARK version:', HARK.__version__)" 2>/dev/null || echo "HARK not available"

# Add activation to bashrc
echo "Adding conda activation to .bashrc..."
echo "conda activate DemARK" >> ~/.bashrc

echo "=== Setup Complete ==="
echo "Environment should be ready to use!" 
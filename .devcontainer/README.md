# DemARK Development Container

This directory contains the configuration for a VS Code development container that provides a consistent, reproducible environment for working with DemARK notebooks and investigating CI/caching issues.

## 🚀 Quick Start

### Prerequisites
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/get-started)

### Getting Started
1. Open this repository in VS Code
2. When prompted, click "Reopen in Container" or use Command Palette → "Dev Containers: Reopen in Container"
3. Wait for the container to build (first time takes ~5-10 minutes)
4. The post-create script will automatically set up the environment

## 🛠️ What's Included

### Environment
- **Base**: Micromamba (lightweight conda alternative)
- **Python Environment**: Created from `binder/environment.yml`
- **HARK**: Installed from master branch (configurable)
- **Jupyter**: Full JupyterLab environment with extensions

### Development Tools
- **Python**: Black formatter, Ruff linter, Pylance language server
- **Jupyter**: Full notebook support with widgets
- **Git**: Pre-configured with GitHub CLI
- **Diagnostic Tools**: All DemARK analysis scripts ready to use

### VS Code Extensions
- Python development (Python, Pylance, Black, Ruff)
- Jupyter notebook support
- Markdown and YAML editing
- JSON formatting

## 📋 Common Tasks

### Run Jupyter Lab
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Access at: http://localhost:8888

### Test Notebooks
```bash
# Test all notebooks (excluding problematic ones)
python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
  --ignore=notebooks/Chinese-Growth.ipynb \
  --ignore=notebooks/Harmenberg-Aggregation.ipynb \
  notebooks/

# Test specific notebook
python -m pytest --nbval-lax notebooks/Gentle-Intro-To-HARK-PerfForesightCRRA.ipynb
```

### Investigate CI Caching Issues
```bash
# Run the bisection analysis
./bisect_hark_breaking_changes.sh

# Analyze caching problems
cd caching_problems_fix
./reproduce-problematic-combo.sh
```

### Work with Historical Versions
```bash
# Compare different DemARK versions
cd DemARK_20250628-2309_current   # Current working version
cd DemARK_20231129-1727_history   # Historical version
cd DemARK_20240918-0003_counter   # Counterfactual version
```

## 🔧 Environment Configuration

### Conda Environment
The container uses the exact same `binder/environment.yml` as the main repository, ensuring consistency with CI and Binder environments.

### HARK Version Testing
To test different HARK versions:
```bash
# Install specific HARK version
pip install git+https://github.com/econ-ark/HARK@v0.16.1

# Or install from specific commit
pip install git+https://github.com/econ-ark/HARK@7a6e8f39
```

### Environment Variables
- `MAMBA_ROOT_PREFIX`: Points to micromamba installation
- `PATH`: Includes conda environment binaries
- `SHELL`: Set to bash for consistency

## 🐛 Troubleshooting

### Container Won't Build
- Check Docker is running
- Ensure you have enough disk space (~2GB for container)
- Try rebuilding: Command Palette → "Dev Containers: Rebuild Container"

### HARK Import Issues
This is expected when testing caching problems! The container can install different HARK versions for testing.

### Jupyter Not Starting
```bash
# Manually activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate DemARK

# Check jupyter installation
jupyter --version
```

### Permission Issues
The container runs as the `vscode` user with sudo privileges:
```bash
sudo apt-get update  # Install system packages
```

## 🔍 Debugging CI Issues

This container is specifically designed to help debug the CI caching issues:

### Reproduce CI Environment
```bash
# The container mimics the CI environment
# Use the diagnostic scripts to investigate issues
./bisect_hark_breaking_changes.sh --help
```

### Test Different Scenarios
```bash
# Test with cached environment (historical)
cd DemARK_20231129-1727_history
./run_tests.sh

# Test with fresh environment (current)
cd DemARK_20250628-2309_current
./run_tests.sh
```

## 📁 File Structure

```
.devcontainer/
├── devcontainer.json    # Main configuration
├── Dockerfile          # Container definition
├── post-create.sh      # Setup script
└── README.md          # This file
```

## 🤝 Contributing

When making changes to the devcontainer:
1. Test the changes by rebuilding the container
2. Update this README if adding new features
3. Ensure the container works for both development and CI investigation

## 📚 Additional Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [DemARK Repository](https://github.com/econ-ark/DemARK) 
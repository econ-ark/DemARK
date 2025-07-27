# DemARK Development Container

This directory contains the configuration for a Visual Studio Code development container that provides a consistent development environment for the DemARK project.

## Prerequisites

### Required Software

- **Docker Desktop**: Version 4.10.0+ recommended (tested with v25.0.3)
    - [Install Docker Desktop](https://www.docker.com/get-started)
    - Ensure Docker is running with at least 4GB RAM allocated
    - Enable "Use Docker Compose V2" in Docker Desktop settings
- **Dev Container CLI** (for command-line usage): `npm install -g @devcontainers/cli`

### IDE Options

**Option A: Visual Studio Code (Recommended)**

- [Visual Studio Code](https://code.visualstudio.com/) v1.75.0+
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) v0.327.0+

**Option B: Cursor IDE**  

- [Cursor](https://cursor.sh/) with native devcontainer support
- Note: Auto-detection may not work reliably; use manual connection methods

**Option C: Command Line Only**

- Use the DevContainer CLI directly (see CLI Usage section below)

## Getting Started

### Method 1: Visual Studio Code (Automatic)

1. **Open the repository in VS Code**

   ```bash
   code /path/to/DemARK
   ```

2. **Reopen in Container**
   - VS Code should automatically detect the devcontainer configuration and show a notification
   - Click "Reopen in Container" when prompted
   - Alternatively, use the Command Palette (Ctrl+Shift+P / Cmd+Shift+P) and select "Dev Containers: Reopen in Container"

### Method 2: Command Line Interface

```bash
# Build and start the devcontainer
devcontainer up --workspace-folder .

# Connect to the running container
devcontainer exec --workspace-folder . bash

# You'll see the prompt: (DemARK) vscode ➜ /workspaces/DemARK
# The environment is now ready to use!
```

### Method 3: Manual Connection (for Cursor or troubleshooting)

```bash
# Build the container
devcontainer build --workspace-folder .

# Get container ID
docker ps

# Connect directly
docker exec -it <container-id> bash
```

### First-time Setup (All Methods)

- The container will automatically build and install all dependencies from `binder/environment.yml`
- This process may take 5-15 minutes the first time depending on your internet connection
- The conda environment "DemARK" will be created and activated automatically

## What's Included

### Environment

- **Base Image**: Microsoft's miniconda devcontainer
- **Python**: 3.10 (as specified in environment.yml)
- **Conda Environment**: Automatically created from `binder/environment.yml`
- **Dependencies**: All packages from the environment.yml including:
    - Scientific computing: numpy, scipy, pandas, matplotlib, seaborn
    - Economics: HARK library from econ-ark
    - Jupyter ecosystem: jupyter, jupyterlab, notebook, ipywidgets for interactive notebooks
    - Statistical modeling: statsmodels, linearmodels

### VS Code Extensions

- **Python Development**: Python, Pylance, Black formatter, Flake8
- **Jupyter**: Full Jupyter notebook support with cell tags and slideshow
- **Markdown**: Enhanced markdown editing for documentation
- **Git**: Built-in git support with GitHub CLI
- **Live Share**: Collaborative development support

### Port Forwarding

- **8888**: Jupyter Lab/Notebook server (default)
- **8889**: Alternative Jupyter port (if 8888 is in use)

## Using the Environment

### Running Jupyter Notebooks

```bash
# The DemARK conda environment is automatically activated
# Option 1: Use Jupyter Lab (recommended for web interface)
jupyter lab --ip=0.0.0.0 --no-browser

# Option 2: Use classic Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --no-browser

# Option 3: Use VS Code's built-in Jupyter support (recommended for IDE)
# Just open any .ipynb file in VS Code - no server needed
```

### Running CI Tests Locally

Validate all notebooks with the same tests used in GitHub Actions:

```bash
# Run the full test suite (matches CI exactly)
python -m pytest --nbval-lax --nbval-cell-timeout=12000 \
    --ignore=notebooks/Chinese-Growth.ipynb \
    --ignore=notebooks/Harmenberg-Aggregation.ipynb \
    notebooks/

# Quick test of a single notebook
python -m pytest --nbval-lax notebooks/Alternative-Combos-Of-Parameter-Values.ipynb
```

### Running Python Scripts

```bash
# Python interpreter is pre-configured with all dependencies
python your_script.py

# Check HARK installation
python -c "import HARK; print(f'HARK version: {HARK.__version__}')"
```

### Managing Dependencies

If you need to add new packages:

```bash
# Add to binder/environment.yml, then:
conda env update -f binder/environment.yml

# Or for development dependencies:
pip install package-name
```

### Command Line Tips

```bash
# Check environment status
conda info --envs

# List installed packages  
conda list

# Activate environment (should be automatic)
conda activate DemARK
```

## Troubleshooting

### Docker Issues

- **Ensure Docker is running** with at least 4GB RAM allocated
- **For macOS/Windows**: Use Docker Desktop v4.10.0+ for best compatibility
- **For Linux**: Ensure docker daemon is running: `sudo systemctl start docker`
- **Storage**: Ensure sufficient disk space (container needs ~2-3GB)

### Container Build Issues

- Try rebuilding: Command Palette → "Dev Containers: Rebuild Container"
- Force rebuild from CLI: `devcontainer build --workspace-folder . --no-cache`
- Check Docker logs: `docker logs <container-id>`

### IDE Connection Issues

- **VS Code**: Auto-detection should work if Dev Containers extension is installed
- **Cursor**: May not auto-detect; use Command Palette → search for "Remote" or "Container"
- **Manual connection**: Use CLI method if IDE integration fails

### Python/Environment Issues

- Python interpreter should be: `/opt/conda/envs/DemARK/bin/python`
- If environment not activated: `conda activate DemARK`  
- Missing packages: `conda env update -f binder/environment.yml`

### Permission Issues

- Git safe directory: Already configured in container
- File permissions: Container runs as `vscode` user, files should be accessible
- Docker socket: Ensure your user is in `docker` group (Linux)

## Advanced Usage

### Customizing the Environment

- Edit `.devcontainer/devcontainer.json` to modify the configuration
- Add additional VS Code extensions in the `extensions` array
- Modify the `postCreateCommand` to run additional setup scripts

### Accessing the Container Shell

- Use VS Code's integrated terminal (automatically opens in the container)
- Or use Command Palette → "Dev Containers: Open Container Configuration File"

# DemARK Development Container

This directory contains the configuration for a Visual Studio Code development container that provides a consistent development environment for the DemARK project.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed and running
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code

## Getting Started

1. **Open the repository in VS Code**
   ```bash
   code /path/to/DemARK
   ```

2. **Reopen in Container**
   - VS Code should automatically detect the devcontainer configuration and show a notification
   - Click "Reopen in Container" when prompted
   - Alternatively, use the Command Palette (Ctrl+Shift+P / Cmd+Shift+P) and select "Dev Containers: Reopen in Container"

3. **First-time Setup**
   - The container will automatically build and install all dependencies from `binder/environment.yml`
   - This process may take several minutes the first time
   - The conda environment "DemARK" will be created and activated

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
# Option 1: Use Jupyter Lab (recommended)
jupyter lab

# Option 2: Use classic Jupyter Notebook
jupyter notebook

# Option 3: Use VS Code's built-in Jupyter support
# Just open any .ipynb file in VS Code - no server needed
```

### Running Python Scripts
```bash
# Python interpreter is pre-configured
python your_script.py
```

### Managing Dependencies
If you need to add new packages:
```bash
# Add to environment.yml, then rebuild the environment
conda env update -f binder/environment.yml
```

## Troubleshooting

### Container Build Issues
- Ensure Docker is running and has sufficient resources
- Try rebuilding the container: Command Palette → "Dev Containers: Rebuild Container"

### Python Path Issues
- The Python interpreter should automatically be set to `/opt/conda/envs/DemARK/bin/python`
- If not, manually select it in VS Code's Python interpreter picker

### Git Configuration
- The container automatically adds the workspace to git's safe directories
- Your local git configuration should be available in the container

## Advanced Usage

### Customizing the Environment
- Edit `.devcontainer/devcontainer.json` to modify the configuration
- Add additional VS Code extensions in the `extensions` array
- Modify the `postCreateCommand` to run additional setup scripts

### Accessing the Container Shell
- Use VS Code's integrated terminal (automatically opens in the container)
- Or use Command Palette → "Dev Containers: Open Container Configuration File" 
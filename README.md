# DemARK

Demonstrations of how to use material in the [Econ-ARK](https://github.com/econ-ark/HARK).

[![launch Binder (main branch)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)
[![launch Binder (stable release)](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/0.13.0)

[![DemARK build on MacOS, Ubuntu and Windows](https://github.com/econ-ark/DemARK/actions/workflows/build.yml/badge.svg)](https://github.com/econ-ark/DemARK/actions/workflows/build.yml)

## Local installation

### Option 1: With Jupyter

1. [Install Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Clone `DemARK` to the folder of your choice
3. Using conda, install the environment\
    `conda env create -f binder/environment.yml`
    - If you already have installed the `DemARK` environment, you may still need to update it\
        `conda env update -f binder/environment.yml`
4. Activate your `DemARK` environment:\
    `conda activate DemARK`
5. Install JupyterLab in the `DemARK` environment:\
    `conda install jupyterlab`
6. Run `jupyter lab` from the `DemARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
7. Run the notebook by choosing `Kernel → Restart & Run All`

### Option 2: With Docker and repo2docker

0. [Install Docker](https://www.docker.com/community-edition)
1. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation), using the "install from source" instructions
2. Run `jupyter repo2docker https://github.com/econ-ark/DemARK`
3. Follow the link in your terminal to the running instance of jupyter
4. Run the notebook by choosing `Kernel → Restart & Run All`

### Option 3: With Development Container

For a consistent development environment with VS Code integration:

0. **Requirements**: Docker Desktop 4.10+ and VS Code with Dev Containers extension
1. Clone the repository and open in VS Code
2. Click "Reopen in Container" when prompted (or use Command Palette → "Dev Containers: Reopen in Container")
3. Container automatically builds with the complete DemARK environment

**Alternative CLI usage:**

```bash
# Build and run the devcontainer
devcontainer up --workspace-folder .

# Connect to running container
devcontainer exec --workspace-folder . bash

# Run notebooks in container
devcontainer exec --workspace-folder . jupyter lab --ip=0.0.0.0 --no-browser
```

See [`.devcontainer/README.md`](.devcontainer/README.md) for detailed setup instructions.

## Contributions

We are eager to encourage contributions.

These can take the form either of new notebooks, or proposed edits for existing notebooks. Either kind of contribution can be made by issuing a pull request.

## Issues

Open an issue in this repository!

## Trigger a test on demand

If you have the proper permissions and want to test whether the DemARKs work with the latest development version of HARK,

[click on the last workflow run here](https://github.com/econ-ark/DemARK/actions/workflows/build.yml) and click the **Re-run all jobs** button

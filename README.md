# DemARK

Demonstrations of how to use material in the [Econ-ARK](https://github.com/econ-ark/HARK).

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/main)

[![DemARK build on MacOS, Ubuntu and Windows](https://github.com/econ-ark/DemARK/actions/workflows/build.yml/badge.svg)](https://github.com/econ-ark/DemARK/actions/workflows/build.yml)

## Local installation

### Option 1: With uv (recommended)

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone `DemARK` to the folder of your choice
3. Install dependencies: `uv sync --extra dev`
4. Run JupyterLab: `uv run jupyter lab`
5. Run the notebook by choosing `Kernel → Restart & Run All`

### Option 2: With conda

1. [Install Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Clone `DemARK` to the folder of your choice
3. Using conda, install the environment:
   `conda env create -f binder/environment.yml`
   - If you already have installed the `DemARK` environment, you may still need to update it:
     `conda env update -f binder/environment.yml`
4. Activate your `DemARK` environment: `conda activate DemARK`
5. Install JupyterLab in the `DemARK` environment: `conda install jupyterlab`
6. Run `jupyter lab` from the `DemARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
7. Run the notebook by choosing `Kernel → Restart & Run All`

### Option 3: With Docker and repo2docker

0. [Install Docker](https://www.docker.com/community-edition)
1. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation), using the "install from source" instructions
2. Run `jupyter repo2docker https://github.com/econ-ark/DemARK`
3. Follow the link in your terminal to the running instance of jupyter
4. Run the notebook by choosing `Kernel → Restart & Run All`

## Contributions

We are eager to encourage contributions.

These can take the form either of new notebooks, or proposed edits for existing notebooks. Either kind of contribution can be made by issuing a pull request.

## Issues

Open an issue in this repository!

## Trigger a test on demand

If you have the proper permissions and want to test whether the DemARKs work with the latest development version of HARK, 

[click on the last workflow run here](https://github.com/econ-ark/DemARK/actions/workflows/build.yml) and click the **Re-run all jobs** button

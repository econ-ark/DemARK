# DemARK

Twenty-two Jupyter notebooks that demonstrate [Econ-ARK](https://github.com/econ-ark/HARK), building
consumption and saving models one assumption at a time, from a two-period problem to a life cycle
with uninsurable risk. Read them at [econ-ark.github.io/DemARK](https://econ-ark.github.io/DemARK),
where each one runs in the browser, or install them below and run them locally.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/main)
[![DemARK build on MacOS, Ubuntu and Windows](https://github.com/econ-ark/DemARK/actions/workflows/build.yml/badge.svg)](https://github.com/econ-ark/DemARK/actions/workflows/build.yml)

Every notebook is meant to be run and edited. They need Python 3.12 or later.

## Install and run locally

### With uv

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository
3. `uv sync --extra dev`
4. `uv run jupyter lab`
5. Open a notebook and choose `Kernel -> Restart & Run All`

The `dev` extra brings in the released `econ-ark` from PyPI along with JupyterLab and the test
tools. It is the quickest path.

### With conda

1. [Install Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
2. Clone this repository
3. `conda env create -f binder/environment.yml`, or `conda env update -f binder/environment.yml` if
   you already have a `DemARK` environment
4. `conda activate DemARK`
5. `conda install jupyterlab`
6. Run `jupyter lab` from the repository root
7. Open a notebook and choose `Kernel -> Restart & Run All`

`binder/environment.yml` installs HARK from its `main` branch, so this path, Binder and the build
badge above all exercise the notebooks against HARK under development. Where a notebook behaves
differently across the two install paths, the uv one is the released behaviour.

### With Docker and repo2docker

1. [Install Docker](https://www.docker.com/community-edition)
2. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation) with its
   install-from-source instructions
3. `jupyter repo2docker https://github.com/econ-ark/DemARK`
4. Follow the link your terminal prints to the running Jupyter instance
5. Open a notebook and choose `Kernel -> Restart & Run All`

## Contributing

New notebooks and edits to existing ones are equally welcome, both as pull requests. A new notebook
should run top to bottom from a fresh kernel, since that is what CI checks and what a reader will
do first. For anything that is not a change to the notebooks, please
[open an issue](https://github.com/econ-ark/DemARK/issues).

## Running the build on demand

With write access to this repository, open the
[most recent build run](https://github.com/econ-ark/DemARK/actions/workflows/build.yml) and press
**Re-run all jobs**. This checks every notebook against the current development version of HARK,
which is how a break in HARK `main` is caught here before a release.

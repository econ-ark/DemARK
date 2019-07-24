# About DemARK

## Repo Layout

The root of the DemARK repo contains the following:

**Directories**

- `binder` contains configuration for the mybinder.org service
- `docs` contains docs about the repo
- `lib` contains utilities such as progress bars
- `notebooks` contains Jupyter notebooks and Python files for different topics

**Files**

- README.md
- `environment.yml`: for local conda install
- `requirements.txt`: for local pip install
- `tasks.py`: configuration for invoke
- `talks.yml`: allows a repo to be configured for a particular workshop or talk. See JupyterLab demo repo for more info.

## Binder configuration

- The `binder` directory contains configuration for the mybinder.org service.
- If a `binder` directory is not present, mybinder.org will default to using the `environment.yml` file when launching.
- The `binder` configuration includes several files:
    - `environment.yml` conda and conda-forge config for repo on binder service
    - `postBuid` commands that are executed by Binder after an initial environment is set up


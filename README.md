# DemArk

This repository contains demos and documentation for [HARK](https://github.com/econ-ark/HARK).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)

## Local installation

*Warning:* DemARK requires Python 2.7 and will not run under Python 3.

### Option 1: With Jupyter

0. [Install jupyter](https://jupyter.org/install).
1. Clone `DemArk` to the folder of your choice
2. Run `pip install -r binder/requirements.txt` to install dependencies
3. Run `binder/postBuild` to enable notebook extensions
4. Run `jupyter notebook` from the `DemARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
5. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

### Option 2: With Docker and repo2docker

0. [Install Docker](https://www.docker.com/community-edition)
1. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation), using the "install from source" instructions
2. Run `jupyter repo2docker https://github.com/econ-ark/DemARK`
3. Follow the link in your terminal to the running instance of jupyter
4. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`


## Issues

Open an issue in this repository!

# DemARK

Demonstrations of how to use material in the [Econ-ARK](https://github.com/econ-ark/HARK).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)
[![Build Status](https://travis-ci.org/econ-ark/DemARK.svg?branch=master)](https://travis-ci.org/econ-ark/DemARK)

## Local installation

### Option 1: With Jupyter

0. [Install jupyter](https://jupyter.org/install).
1. Clone `DemARK` to the folder of your choice
2. Run `pip install -r binder/requirements.txt` to install dependencies
3. Enable notebook extensions.

   **On Linux/macOS:**

   Run `binder/postBuild` in your terminal (at a shell in the binder directory, `./postBuild`)

   **On Windows:**

   Run `binder/postBuild.bat`

4. Run `jupyter notebook` from the `DemARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
5. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

### Option 2: With Docker and repo2docker

0. [Install Docker](https://www.docker.com/community-edition)
1. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation), using the "install from source" instructions
2. Run `jupyter repo2docker https://github.com/econ-ark/DemARK`
3. Follow the link in your terminal to the running instance of jupyter
4. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

## Contributions

We are eager to encourage contributions.

These can take the form either of new notebooks, or proposed edits for existing notebooks. Either kind of contribution can be made by issuing a pull request.

However, to deal with the well-known problem that normal jupyter notebooks do not "play nicely" with github version control, we will require interactions 
with contributors to be conducted after the installation of the [jupytext](https://towardsdatascience.com/introducing-jupytext-9234fdff6c57) tool.
Specifically, you will need to follow the instructions for installing jupytext on your computer, and then need to configure it to use the "percent"
format. Over time, we intend to add the necessary metadata to all our jupyter notebooks to make them automatically invoke jupytext when compiled.

## Issues

Open an issue in this repository!

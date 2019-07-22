# DemARK

This repository contains demos and documentation for [HARK](https://github.com/econ-ark/HARK).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)

## Try the notebooks on Binder

**No installation is needed. Runs in a web browser**

Click the Binder link. Binder builds an environment in the cloud for you to use.
The first time Binder launches it may take a while since it is busy creating a
container to run the demo notebooks. Subsequent launches should be quicker.

TODO add binder link and specify a SHA1

## Local installation

The installation instructions are provided for Python 3.6+.

### Option 1a: With Jupyter using pip

1. Create a virtual environment (using the CPython `venv` command).
2. Activate the `myarkenv` environment.
3. Upgrade pip.
4. Use pip to install requirements.
5. Run Jupyter notebook.

```
python3 -m venv myarkenv
source myarkenv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt
jupyter notebook
```

This will launch the jupyter file browser. The notebooks can be selected and
run.


### Option 1b: With Jupyter using conda

TODO: create an environment.yml

### Option 2

TODO Create a docker container and host on Docker Hub.

---

### Option 1: With Jupyter

1. [Install jupyter](https://jupyter.org/install).
2. Clone `DemARK` to the folder of your choice
3. Run `pip install -r binder/requirements.txt` to install dependencies
4. Enable notebook extensions.

   **On Linux/macOS:**

   Run `binder/postBuild` in your terminal (at a shell in the binder directory, `./postBuild`)

   **On Windows:**

   Run `binder/postBuild.bat`

5. Run `jupyter notebook` from the `DemARK` root folder. You will be prompted to open a page in your web browser. From there, you will be able to run the notebooks.
6. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

### Option 2: With Docker and repo2docker

0. [Install Docker](https://www.docker.com/community-edition)
1. [Install `repo2docker`](https://github.com/jupyter/repo2docker#installation), using the "install from source" instructions
2. Run `jupyter repo2docker https://github.com/econ-ark/DemARK`
3. Follow the link in your terminal to the running instance of jupyter
4. Run the notebook by clicking the `▶▶` button or choosing `Kernel → Restart & Run All`

---

## Contributions

We are eager to encourage contributions.

These can take the form either of new notebooks, or proposed edits for existing notebooks. Either kind of contribution can be made by issuing a pull request.

However, to deal with the well-known problem that normal jupyter notebooks do not "play nicely" with github version control, we will require interactions
with contributors to be conducted after the installation of the [jupytext](https://towardsdatascience.com/introducing-jupytext-9234fdff6c57) tool.
Specifically, you will need to follow the instructions for installing jupytext on your computer, and then need to configure it to use the "percent"
format. Over time, we intend to add the necessary metadata to all our jupyter notebooks to make them automatically invoke jupytext when compiled.

## Issues

Open an issue in this repository!

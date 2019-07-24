# DemARK

This repository contains demos and documentation for [HARK](https://github.com/econ-ark/HARK).

## Try the notebooks on Binder

**No installation is needed. Runs in a web browser**

Click the Binder link. Binder builds an environment in the cloud for you to use.
The first time Binder launches it may take a while since it is busy creating a
container to run the demo notebooks. Subsequent launches should be quicker.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)


## Local installation

The installation instructions are provided for Python 3.6+.

### Option 1a: With Jupyter using pip

1. Create a virtual environment (using the CPython `venv` command).
2. Activate the `myarkenv` environment.
3. Upgrade pip.
4. Use pip to install requirements.
5. Add and enable notebook extensions
6. Run Jupyter notebook.

```
python3 -m venv myarkenv
source myarkenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter contrib nbextension install --user  # installs css/js and nbconvert config info
jupyter nbextension enable codefolding/main  # enable codefolding in a notebook code cell
jupyter nbextension enable codefolding/edit  # enable codefolding in editor
jupyter nbextension enable --py latex_envs  # enable some latex features into notebook
python3 -m cite2c.install  # Enables the cite2c extension (you will need to log into zotero if you use this extension) **Optional**
jupyter notebook
```

This will launch the jupyter file browser. The notebooks can be selected and
run.

Locally, you can enable/disable extensions by: http://localhost:8888/nbextensions. More information can be found in the [notebook extensions documentation](https://jupyter-contrib-nbextensions.readthedocs.io)

---

### Option 1b: With Jupyter using conda

Using conda from Anaconda or miniconda, enter the following to create a local
conda environment.

```
conda env create -f environment.yml
conda activate demos
jupyter contrib nbextension install --user  # installs css/js and nbconvert config info
jupyter nbextension enable codefolding/main  # enable codefolding in a notebook code cell
jupyter nbextension enable codefolding/edit  # enable codefolding in editor
jupyter nbextension enable --py latex_envs  # enable some latex features into notebook
python3 -m cite2c.install  # Enables the cite2c extension (you will need to log into zotero if you use this extension) **Optional**
jupyter notebook
```

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

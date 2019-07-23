# DemARK

This repository contains demos and documentation for [HARK](https://github.com/econ-ark/HARK).

## Try the notebooks on Binder

**No installation is needed. Runs in a web browser**

Click the Binder link. Binder builds an environment in the cloud for you to use.
The first time Binder launches it may take a while since it is busy creating a
container to run the demo notebooks. Subsequent launches should be quicker.

**Use this link while testing**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/willingc/DemARK/troubleshoot)


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
pip install -r requirements-local.txt
jupyter notebook
```

This will launch the jupyter file browser. The notebooks can be selected and
run.

You may also wish to activate a few notebook extensions for convenience.
- First, stop Jupyter Notebook.
- Next execute the following at the command line:

```
# Docs on nbextensions: https://jupyter-contrib-nbextensions.readthedocs.io
jupyter contrib nbextension install --user  # installs css/js and nbconvert config info
jupyter nbextension enable codefolding/main  # enable codefolding in a notebook code cell
jupyter nbextension enable codefolding/edit  # enable codefolding in editor
jupyter nbextension enable --py latex_envs  # enable some latex features into notebook
python3 -m cite2c.install  # Enables the cite2c extension (you will need to log into zotero if you use this extension) **Optional**
jupyter notebook
```

Locally, you can enable/disable extensions by: http://localhost:8888/nbextensions

---

### Option 1b: With Jupyter using conda

TODO: create an environment.yml

### Option 2

TODO Create a docker container and host on Docker Hub.

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

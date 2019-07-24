# To Do list

## Security

- hash or hide zotero credential token

## Enhancements for Binder or Local use
- Add invoke to simplify for local user setup (similar to jupyterlab demo)
- Use environment.yml and conda-forge in Binder
- To Pin or not to pin requirements

## Tests

- Add basic tests for running notebooks success (Papermill or invoke)

## Notebook issues

- Check MyType error in Alternative-Combos-Of-Parameter-Values.ipynb

### Latex Errors

Latex errors Gentle-Intro-To-HARK-Buffer-Stock-Model nb

### Runtime or module errors (version or logic error?)
- /srv/conda/envs/notebook/lib/python3.7/site-packages/HARK/interpolation.py:1675: RuntimeWarning: All-NaN slice encountered
- LifecycleModelExample nb error No module named 'HARK.SolvingMicroDSOPs.EstimationParameters'

### Can't find data file

- MicroMacroProblems Both nbs error at Lorenz cell FileNotFoundError: [Errno 2] No such file or directory: '/srv/conda/envs/notebook/lib/python3.7/site-packages/HARK/cstwMPC/USactuarial.txt'
- StructuralEstimates nb error FileNotFoundError: [Errno 2] No such file or directory: '/srv/conda/envs/notebook/lib/python3.7/site-packages/HARK/cstwMPC/USactuarial.txt'
- Uncertainty-and-Savings-Rate nb No such file or directory: '/srv/conda/envs/notebook/lib/python3.7/site-packages/HARK/cstwMPC/USactuarial.txt'


### Slow, Runs until student exercise

- Check heading click for DCEGM-Upper-Envelope nb
- Check last cell execution Fashion-Victim-Model nb (It works but serializes execution) Slow
- KrusellSmith.ipynb Solver is slow...
- IncExpectation notebooks are slow...
- Runs until student exercise: Gentle-Intro-To-HARK-PerfForesightCRRA
- Runs until student exercise: Latex?? Gentle-Intro-To-HARK
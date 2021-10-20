# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.8
# ---

# %% [markdown]
# # Perfect Foresight CRRA Model - Saving Rate
#
# [![badge](https://img.shields.io/badge/Launch%20using%20-Econ--ARK-blue)](https://econ-ark.org/materials/perfforesightcrra-savingrate#launch)
#

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import HARK 
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)
from HARK.utilities import plot_funcs

# These last two will make our charts look nice
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Dark2')

# %%
# Set up a HARK Perfect Foresight Consumer called PFsavrate

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType # Import the consumer type

# Now we need to "fill" our consumer with parameters that allow us to solve the consumer's problem

# First we need to set out a dictionary
CRRA = 2.                           # Coefficient of relative risk aversion
Rfree = 1.03                        # Interest factor on assets
DiscFac = 0.97                      # Intertemporal discount factor
LivPrb = [1.0]                      # Survival probability
PermGroFac = [1.01]                 # Permanent income growth factor
AgentCount = 1                      # Number of agents of this type (only matters for simulation)
T_cycle = 1                         # Number of periods in the cycle for this agent type
cycles = 0                          # Agent is infinitely lived

# Make a dictionary to specify a perfect foresight consumer type
dict_wealth = { 'CRRA': CRRA,
                'Rfree': Rfree,
                'DiscFac': DiscFac,
                'LivPrb': LivPrb,
                'PermGroFac': PermGroFac,
                'AgentCount': AgentCount,
                'T_cycle' : T_cycle,
                'cycles' : cycles
                }

# Now lets pass our dictionary to our consumer class
PFsavrate = PerfForesightConsumerType(**dict_wealth)

# %% [markdown]
# # Now We can plot the saving rate as a function of market resoures

# %%

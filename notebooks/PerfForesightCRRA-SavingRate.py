# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Perfect Foresight CRRA Model - Savings Rate

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
# The first step is to be able to bring things in from different directories
import sys 
import os

sys.path.insert(0, os.path.abspath('../lib'))

import numpy as np
import HARK 
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)
from HARK.utilities import plotFuncs

# These last two will make our charts look nice
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Dark2')

# %% [markdown]
# ## Question 2
# Make some plots that illustrate the points made in sections 4.1 and 4.2 of [PerfForesightCRRA](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Consumption/PerfForesightCRRA.pdf) about the size of the human wealth effect and the relationship between interest rates and the saving rate.

# %% [markdown]
# Firstly, we want to show that for plausible parameter values, the human wealth effect of a fall in interest rate outweighs the income and substition effects, so consumption rises strongly.

# %%
# Set up a HARK Perfect Foresight Consumer called PFwealth

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
PFwealth = PerfForesightConsumerType(**dict_wealth)

# %% [markdown]
# We can see that consumption is higher for all market resources when R is low, owing to the human wealth effect. And that the saving rate is very sensitive to changes in R (look at when m=1, the savings rate goes from -0.1 to- 0.5 when R moves from 1.06 to 1.03.

# %%

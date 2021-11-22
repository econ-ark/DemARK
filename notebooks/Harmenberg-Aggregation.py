# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Preliminaries

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType
)

from HARK.ConsumptionSaving.tests.test_IndShockConsumerType import (
    dict_harmenberg    
)

import numpy as np

# %% [markdown]
# # Description of the problem

# %% [markdown]
# # Description of the method

# %% [markdown]
# # Demonstration of HARK's implementation and the gain in efficiency

# %%

example = IndShockConsumerType(**dict_harmenberg, verbose = 0)
example.cycles = 0
example.track_vars = [ 'aNrm', 'mNrm','cNrm','pLvl','aLvl']
example.T_sim= 20000

example.solve()

example.neutral_measure = True
example.update_income_process()

example.initialize_sim()
example.simulate()




Asset_list = []
Consumption_list = []
M_list =[]


for i in range (example.T_sim):
    Assetagg =  np.mean(example.history['aNrm'][i])
    Asset_list.append(Assetagg)
    ConsAgg =  np.mean(example.history['cNrm'][i] )
    Consumption_list.append(ConsAgg)
    Magg = np.mean(example.history['mNrm'][i])
    M_list.append(Magg)

#########################################################

burnin = 100
sample_every = 50
n_agents = [10,100,1000]

example2 = IndShockConsumerType(**dict_harmenberg, verbose = 0)
example2.cycles = 0
example2.track_vars = [ 'aNrm', 'mNrm','cNrm','pLvl','aLvl']
example2.T_sim= 20000


example2.solve()
example2.initialize_sim()
example2.simulate()

# %% [markdown]
# # Comparison of the PIN-measure and the base measure

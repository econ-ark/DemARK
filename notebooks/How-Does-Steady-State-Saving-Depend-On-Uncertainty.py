# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.3
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
#     version: 3.6.6
#   varInspector:
#     cols:
#       lenName: 16
#       lenType: 16
#       lenVar: 40
#     kernels_config:
#       python:
#         delete_cmd_postfix: ''
#         delete_cmd_prefix: 'del '
#         library: var_list.py
#         varRefreshCmd: print(var_dic_list())
#       r:
#         delete_cmd_postfix: ') '
#         delete_cmd_prefix: rm(
#         library: var_list.r
#         varRefreshCmd: 'cat(var_dic_list()) '
#     types_to_exclude:
#     - module
#     - function
#     - builtin_function_or_method
#     - instance
#     - _Feature
#     window_display: false
# ---

# %% {"code_folding": [0]}
# Initial imports and notebook setup, click arrow to show
%matplotlib inline
# The first step is to be able to bring things in from different directories
import sys 
import os
sys.path.insert(0, os.path.abspath('../lib'))

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from util import log_progress
import HARK # Prevents import error from Demos repo

# %% [markdown]
# # How Does Steady State Saving Depend On Uncertainty?
#

# %% {"code_folding": [0]}
# Initialize the cstwMPC parameters
sav_vs_uncert_pmtrs = {
    "CRRA":1.0,                    # Coefficient of relative risk aversion   
    "Rfree":1.01/(1.0 - 1.0/160.0), # Survival probability,
    "PermGroFac":[1.000**0.25], # Permanent income growth factor (no perm growth),
    "PermGroFacAgg":1.0,
    "BoroCnstArt":0.0,
    "CubicBool":False,
    "vFuncBool":False,
    "PermShkStd":[(0.01*4/11)**0.5],  # Standard deviation of permanent shocks to income
    "PermShkCount":5,  # Number of points in permanent income shock grid
    "TranShkStd":[(0.01*4)**0.5],  # Standard deviation of transitory shocks to income,
    "TranShkCount":5,  # Number of points in transitory income shock grid
    "UnempPrb":0.07,  # Probability of unemployment while working
    "IncUnemp":0.15,  # Unemployment benefit replacement rate
    "UnempPrbRet":None,
    "IncUnempRet":None,
    "aXtraMin":0.00001,  # Minimum end-of-period assets in grid
    "aXtraMax":20,  # Maximum end-of-period assets in grid
    "aXtraCount":20,  # Number of points in assets grid,
    "aXtraExtra":[None],
    "aXtraNestFac":3,  # Number of times to 'exponentially nest' when constructing assets grid
    "LivPrb":[1.0 - 1.0/160.0],  # Survival probability
    "DiscFac":0.97,             # Default intertemporal discount factor, # dummy value, will be overwritten
    "cycles":0,
    "T_cycle":1,
    "T_retire":0,
    'T_sim':1200,  # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
    'T_age': 400,
    'IndL': 10.0/9.0,  # Labor supply per individual (constant),
    'aNrmInitMean':np.log(0.00001),
    'aNrmInitStd':0.0,
    'pLvlInitMean':0.0,
    'pLvlInitStd':0.0,
    'AgentCount':0,  # will be overwritten by parameter distributor
}

# %%
import HARK.cstwMPC.SetupParamsCSTW as Params
import HARK.cstwMPC.cstwMPC as cstwMPC
BaselineType = cstwMPC.cstwMPCagent(**Params.init_infinite)

# %% [markdown]
# Now, add in ex-ante heterogeneity in consumers' discount factors.
#
# The cstwMPC parameters do not define a single discount factor; instead, there is ex-ante heterogeneity in the discount factor.  To prepare to create this ex-ante heterogeneity, first create the desired number of consumer types:
#

# %%
num_consumer_types   = 7 # declare the number of types we want
ConsumerTypes = [] # initialize an empty list

for nn in range(num_consumer_types):
    # Now create the types, and append them to the list ConsumerTypes
    newType = deepcopy(BaselineType)    
    ConsumerTypes.append(newType)

# %% [markdown]
#
# Now, generate the desired ex-ante heterogeneity, by giving the different consumer types each their own discount factor.
#
# First, decide the discount factors to assign:

# %%
from HARK.utilities import approxUniform

bottomDiscFac = 0.9800
topDiscFac    = 0.9934 
DiscFac_list  = approxUniform(N=num_consumer_types,bot=bottomDiscFac,top=topDiscFac)[1]

# Now, assign the discount factors we want to the ConsumerTypes
for j in range(num_consumer_types):
    ConsumerTypes[j].DiscFac = DiscFac_list[j]

# %% [markdown]
# ## Setting Up the Experiment
#
# The experiment is performed by a function we will now write.
#
# Recall that all parameters have been assigned appropriately, except for the income process.
#
# This is because we want to see how much uncertainty needs to accompany the high-growth state to generate the desired high savings rate.
#
# Therefore, among other things, this function will have to initialize and assign the appropriate income process.

# %% {"code_folding": []}
# Create the shock distribution in the low-growth state, which we will not change
from HARK.ConsumptionSaving.ConsIndShockModel import constructLognormalIncomeProcessUnemployment
import HARK.ConsumptionSaving.ConsumerParameters as ShockParams

ShockDstnBase = constructLognormalIncomeProcessUnemployment(ShockParams)[0][0]

# Remember the standard deviation of the permanent income shock in the low-growth state for later
PermShkStd = ShockParams.PermShkStd


def calcNatlSavingRate(PrmShkVar_multiplier,RNG_seed = 0):
    """
    This function actually performs the experiment we want.
    
    Remember this experiment is: get consumers into the steady-state associated with the low-growth
    regime. Then, give them an unanticipated shock that increases the income growth rate
    and permanent income uncertainty at the same time.  What happens to the path for 
    the national saving rate?  Can an increase in permanent income uncertainty
    explain the high  saving rate since economic reforms began?
    
    The inputs are:
        * PrmShkVar_multiplier, the number by which we want to multiply the baseline variance
          of the permanent shock
        * RNG_seed, an integer to seed the random number generator for simulations.  This useful
          because we are going to run this function for different values of PrmShkVar_multiplier,
          and we may not necessarily want the simulated agents in each run to experience
          the same (normalized) shocks.
    """

    # First, make a deepcopy of the ConsumerTypes (each with their own discount factor), 
    # because we are going to alter them
    ConsumerTypesAlt = deepcopy(ConsumerTypes)

    # Set the uncertainty in the alternative state to the desired amount, keeping in mind
    # that PermShkStd is a list of length 1
    PrmShkStd_multiplier = PrmShkVar_multiplier ** .5
    ShkParams.PermShkStd = [BaselineType_PermShkStd[0] * PrmShkStd_multiplier] 

    # Construct the appropriate income distributions
    ShockDstnAlt = constructLognormalIncomeProcessUnemployment(ShkParams)[0][0]

    # To calculate the national saving rate, we need national income and national consumption
    # To get those, we are going to start national income and consumption at 0, and then
    # loop through each agent type and add their values they contribute to accumulate aggregate 
    # income and consumption.
    NatlIncome = 0.
    NatlCons   = 0.

    for ConsumerTypeNew in ConsumerTypesNew:
        ### For each consumer type (i.e. each discount factor), calculate total income 
        ### and consumption

        # First give each ConsumerType their own random number seed
        RNG_seed += 19
        ConsumerTypeNew.seed  = RNG_seed
        
        # Set the income distribution in each Markov state appropriately        
        ConsumerTypeNew.IncomeDstn = [[LowGrowthIncomeDstn,HighGrowthIncomeDstn]]

        # Solve the problem for this ConsumerTypeNew
        ConsumerTypeNew.solve()

        """
        Now we are ready to simulate.
        
        This case will be a bit different than most, because agents' *perceptions* of the probability
        of changes in the  economy will differ from the actual probability of changes.  
        Specifically, agents think there is a 0% chance of moving out of the low-growth state, and 
        that there is a  (1./160) chance of moving out of the high-growth state.  In reality, we 
        want the  economy to reach the low growth steady state, and then move into the 
        high growth state with probability 1.  Then we want it to persist in the high growth 
        state for 40 years. 
        """
        
        ## Now, simulate 500 quarters to get to steady state, then 40 years of high growth
        ConsumerTypeNew.T_sim = 660 
        
        # Ordinarily, the simulate method for a MarkovConsumerType randomly draws Markov states
        # according to the transition probabilities in MrkvArray *independently* for each simulated
        # agent.  In this case, however, we want the discrete state to be *perfectly coordinated*
        # across agents-- it represents a macroeconomic state, not a microeconomic one!  In fact,
        # we don't want a random history at all, but rather a specific, predetermined history: 125
        # years of low growth, followed by 40 years of high growth.
        
        # To do this, we're going to "hack" our consumer type a bit.  First, we set the attribute
        # MrkvPrbsInit so that all of the initial Markov states are in the low growth state.  Then
        # we initialize the simulation and run it for 500 quarters.  However, as we do not
        # want the Markov state to change during this time, we change its MrkvArray to always be in
        # the low growth state with probability 1.
        
        ConsumerTypeNew.MrkvPrbsInit = np.array([1.0,0.0]) # All consumers born in low growth state
        ConsumerTypeNew.MrkvArray[0] = np.array([[1.0,0.0],[1.0,0.0]]) # Stay in low growth state
        ConsumerTypeNew.initializeSim() # Clear the history and make all newborn agents
        ConsumerTypeNew.simulate(500)   # Simulate 500 quarders of data
        
        # Now we want the high growth state to occur for the next 160 periods.  We change the initial
        # Markov probabilities so that any agents born during this time (to replace an agent who
        # died) is born in the high growth state.  Moreover, we change the MrkvArray to *always* be
        # in the high growth state with probability 1.  Then we simulate 160 more quarters.
        
        ConsumerTypeNew.MrkvPrbsInit = np.array([0.0,1.0]) # All consumers born in low growth state
        ConsumerTypeNew.MrkvArray[0] = np.array([[0.0,1.0],[0.0,1.0]]) # Stay in low growth state
        ConsumerTypeNew.simulate(160)   # Simulate 160 quarders of data
    
        # Now, get the aggregate income and consumption of this ConsumerType over time
        IncomeOfThisConsumerType = np.sum((ConsumerTypeNew.aNrmNow_hist*ConsumerTypeNew.pLvlNow_hist*
                                          (ConsumerTypeNew.Rfree[0] - 1.)) +
                                           ConsumerTypeNew.pLvlNow_hist, axis=1)
        
        ConsOfThisConsumerType = np.sum(ConsumerTypeNew.cNrmNow_hist*ConsumerTypeNew.pLvlNow_hist,axis=1)
        
        # Add the income and consumption of this ConsumerType to national income and consumption
        NatlIncome     += IncomeOfThisConsumerType
        NatlCons       += ConsOfThisConsumerType

        
    # After looping through all the ConsumerTypes, calculate and return the path of the national 
    # saving rate
    NatlSavingRate = (NatlIncome - NatlCons)/NatlIncome

    return NatlSavingRate

# %% [markdown]
# Now we can use the function we just defined to calculate the path of the national saving rate following the economic reforms, for a given value of the increase to the variance of permanent income accompanying the reforms.  We are going to graph this path for various values for this increase.
#
# Remember, we want to see if a plausible value for this increase in uncertainty can explain the high  saving rate.

# %%
# Declare the number of periods before the reforms to plot in the graph
quarters_before_reform_to_plot = 5

# Declare the quarters we want to plot results for
quarters_to_plot = np.arange(-quarters_before_reform_to_plot ,160,1)

# Create a list to hold the paths of the national saving rate
NatlSavingsRates = []

# Create a list of floats to multiply the variance of the permanent shock to income by
PermShkVarMultipliers = (1.,2.,4.,8.,11.)

# Loop through the desired multipliers, then get the path of the national saving rate
# following economic reforms, assuming that the variance of the permanent income shock
# was multiplied by the given multiplier
index = 0
for PermShkVarMultiplier in log_progress(PermShkVarMultipliers, every=1):
    NatlSavingsRates.append(calcNatlSavingRate(PermShkVarMultiplier,RNG_seed = index)[-160 - quarters_before_reform_to_plot :])
    index +=1

# %% [markdown]
# We've calculated the path of the national saving rate as we wanted. All that's left is to graph the results!

# %%
plt.ylabel('Natl Saving Rate')
plt.xlabel('Quarters Since Economic Reforms')
plt.plot(quarters_to_plot,NatlSavingsRates[0],label=str(PermShkVarMultipliers[0]) + ' x variance')
plt.plot(quarters_to_plot,NatlSavingsRates[1],label=str(PermShkVarMultipliers[1]) + ' x variance')
plt.plot(quarters_to_plot,NatlSavingsRates[2],label=str(PermShkVarMultipliers[2]) + ' x variance')
plt.plot(quarters_to_plot,NatlSavingsRates[3],label=str(PermShkVarMultipliers[3]) + ' x variance')
plt.plot(quarters_to_plot,NatlSavingsRates[4],label=str(PermShkVarMultipliers[4]) + ' x variance')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
ncol=2, mode="expand", borderaxespad=0.) #put the legend on top
plt.show()

# %% [markdown]
# The figure shows that, if the rate of growth increases the way Chinese growth did, but is not accompanied by any change in the degree of uncertainty, the model's predicted saving rate declines drastically, from an initial (calibrated) value of about 0.1 (ten percent) to close to zero.  For this model to have any hope of predicting an increase in the saving rate, it is clear that the increase in uncertainty that accompanies the increase in growth will have to be substantial.  
#
# The red line shows that a mere doubling of uncertainty from its baseline value is not enough: The steady state saving rate is still below its slow-growth value.
#
# When we assume that the degree of uncertainty quadruples, the model does finally predict that the new steady-state saving rate will be higher than before, but not much higher, and not remotely approaching 25 percent.
#
# Only when the degree of uncertainty increases by a factor of 8 is the model capable of producing a new equilbrium saving rate in the ballpark of the Chinese value.  
#
# But this is getting close to a point where the model starts to break down (for both numerical and conceptual reasons), as shown by the erratic path of the saving rate when we multiply the initial variance by 11.  
#
# We do not have historical data on the magnitude of permanent income shocks in China in the pre-1978 period; it would be remarkable if the degree of uncertainty increased by such a large amount, but in the absence of good data it is hard to know for sure.  
#
# What the experiment does demonstrate, though, is that it is _not_ the case that "it is easy to explain anything by invoking some plausible but unmeasurable change in uncertainty."  Substantial differences in the degree of permanent (or highly persistent) income uncertainty across countries, across periods, and across people have been measured in the literature, and those differences could in principle be compared to differences in saving rates to get a firmer fix on the quantitative importance of the "precautionary saving" explanation in the Chinese context.

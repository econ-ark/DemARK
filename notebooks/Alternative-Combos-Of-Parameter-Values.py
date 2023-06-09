# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
#     version: 3.10.8
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: true
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # Alternative Combinations of Parameter Values
#
# [![badge](https://img.shields.io/badge/Launch%20using%20-Econ--ARK-blue)](https://econ-ark.org/materials/alternative-combos-of-parameter-values#launch)
#
# Please write the names and email addresses of everyone who worked on this notebook on the line below.
#
# YOUR NAMES HERE
#
# ## Introduction
#
# The notebook "Micro-and-Macro-Implications-of-Very-Impatient-HHs" is an exercise that demonstrates the consequences of changing a key parameter of the [cstwMPC](http://www.econ2.jhu.edu/people/ccarroll/papers/cstwMPC) model, the time preference factor $\beta$.
#
# The [REMARK](https://github.com/econ-ark/REMARK) `SolvingMicroDSOPs` reproduces the last figure in the [SolvingMicroDSOPs](http://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs) lecture notes, which shows that there are classes of alternate values of $\beta$ and $\rho$ that fit the data almost as well as the exact 'best fit' combination.
#
# Inspired by this comparison, this notebook asks you to examine the consequences for:
#
# * The consumption function
# * The distribution of wealth
#
# Of _joint_ changes in $\beta$ and $\rho$ together.
#
# One way you can do this is to construct a list of alternative values of $\rho$ (say, values that range upward from the default value of $\rho$, in increments of 0.2, all the way to $\rho=5$).  Then for each of these values of $\rho$ you will find the value of $\beta$ that leads the same value for target market resources, $\check{m}$.
#
# As a reminder, $\check{m}$ is defined as the value of $m$ at which the optimal value of ${c}$ is the value such that, at that value of ${c}$, the expected level of ${m}$ next period is the same as its current value:
#
# $\mathbb{E}_{t}[{m}_{t+1}] = {m}_{t}$
#
# Other notes:
# * The cstwMPC model solves and simulates the problems of consumers with 7 different values of $\beta$
#    * You should do your exercise using the middle value of $\beta$ from that exercise:
#       * `DiscFac_mean   = 0.9855583`
# * You are likely to run into the problem, as you experiment with parameter values, that you have asked HARK to solve a model that does not satisfy one of the impatience conditions required for the model to have a solution.  Those conditions are explained intuitively in the [TractableBufferStock](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/TractableBufferStock/) model.  The versions of the impatience conditions that apply to the $\texttt{IndShockConsumerType}$ model can be found in the paper [BufferStockTheory](http://www.econ2.jhu.edu/people/ccarroll/papers/BufferStockTheory), table 2.
#    * The conditions that need to be satisfied are:
#       * The Growth Impatience Condition (GIC)
#       * The Return Impatience Condition (RIC)
# * Please accumulate the list of solved consumers' problems in a list called `MyTypes`
#    * For compatibility with a further part of the assignment below

# %% {"code_folding": []}
# This cell merely imports and sets up some basic functions and packages

# %matplotlib inline
from HARK.utilities import get_lorenz_shares, get_percentiles
from tqdm import tqdm
import numpy as np


# %% {"code_folding": [0, 4]}
# Import IndShockConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

# Define a dictionary with calibrated parameters
cstwMPC_calibrated_parameters = {
    "CRRA": 1.0,  # Coefficient of relative risk aversion
    "Rfree": 1.01 / (1.0 - 1.0 / 160.0),  # Survival probability,
    # Permanent income growth factor (no perm growth),
    "PermGroFac": [1.000**0.25],
    "PermGroFacAgg": 1.0,
    "BoroCnstArt": 0.0,
    "CubicBool": False,
    "vFuncBool": False,
    "PermShkStd": [
        (0.01 * 4 / 11) ** 0.5
    ],  # Standard deviation of permanent shocks to income
    "PermShkCount": 5,  # Number of points in permanent income shock grid
    "TranShkStd": [
        (0.01 * 4) ** 0.5
    ],  # Standard deviation of transitory shocks to income,
    "TranShkCount": 5,  # Number of points in transitory income shock grid
    "UnempPrb": 0.07,  # Probability of unemployment while working
    "IncUnemp": 0.15,  # Unemployment benefit replacement rate
    "UnempPrbRet": 0.0,
    "IncUnempRet": 0.0,
    "aXtraMin": 0.00001,  # Minimum end-of-period assets in grid
    "aXtraMax": 40,  # Maximum end-of-period assets in grid
    "aXtraCount": 32,  # Number of points in assets grid
    "aXtraExtra": [None],
    "aXtraNestFac": 3,  # Number of times to 'exponentially nest' when constructing assets grid
    "LivPrb": [1.0 - 1.0 / 160.0],  # Survival probability
    "DiscFac": 0.97,  # Default intertemporal discount factor; dummy value, will be overwritten
    "cycles": 0,
    "T_cycle": 1,
    "T_retire": 0,
    # Number of periods to simulate (idiosyncratic shocks model, perpetual youth)
    "T_sim": 1200,
    "T_age": 400,
    "IndL": 10.0 / 9.0,  # Labor supply per individual (constant),
    "aNrmInitMean": np.log(0.00001),
    "aNrmInitStd": 0.0,
    "pLvlInitMean": 0.0,
    "pLvlInitStd": 0.0,
    "AgentCount": 10000,
}

# %%
# Construct a list of solved consumers' problems, IndShockConsumerType is just a place holder
MyTypes = [IndShockConsumerType(verbose=0, **cstwMPC_calibrated_parameters)]

# %% [markdown]
# ## Simulating the Distribution of Wealth for Alternative Combinations
#
# You should now have constructed a list of consumer types all of whom have the same _target_ level of market resources $\check{m}$.
#
# But the fact that everyone has the same target ${m}$ does not mean that the _distribution_ of ${m}$ will be the same for all of these consumer types.
#
# In the code block below, fill in the contents of the loop to solve and simulate each agent type for many periods.  To do this, you should invoke the methods $\texttt{solve}$, $\texttt{initialize_sim}$, and $\texttt{simulate}$ in that order.  Simulating for 1200 quarters (300 years) will approximate the long run distribution of wealth in the population.

# %%
for ThisType in tqdm(MyTypes):
    ThisType.solve()
    ThisType.initialize_sim()
    ThisType.simulate()

# %% [markdown]
# Now that you have solved and simulated these consumers, make a plot that shows the relationship between your alternative values of $\rho$ and the mean level of assets

# %%
# To help you out, we have given you the command needed to construct a list of the levels of assets for all consumers
aLvl_all = np.concatenate([ThisType.state_now["aLvl"] for ThisType in MyTypes])

# You should take the mean of aLvl for each consumer in MyTypes, divide it by the mean across all simulations
# and then plot the ratio of the values of mean(aLvl) for each group against the value of $\rho$

# %% [markdown]
# # Interpret
# Here, you should attempt to give an intiutive explanation of the results you see in the figure you just constructed

# %% [markdown]
# ## The Distribution of Wealth...
#
# Your next exercise is to show how the distribution of wealth differs for the different parameter  values

# %%
# Finish filling in this function to calculate the Euclidean distance between the simulated and actual Lorenz curves.


def calcLorenzDistance(SomeTypes):
    """
    Calculates the Euclidean distance between the simulated and actual (from SCF data) Lorenz curves at the
    20th, 40th, 60th, and 80th percentiles.

    Parameters
    ----------
    SomeTypes : [AgentType]
        List of AgentTypes that have been solved and simulated.  Current levels of individual assets should
        be stored in the attribute aLvl.

    Returns
    -------
    lorenz_distance : float
        Euclidean distance (square root of sum of squared differences) between simulated and actual Lorenz curves.
    """
    # Define empirical Lorenz curve points
    lorenz_SCF = np.array([-0.00183091, 0.0104425, 0.0552605, 0.1751907])

    # Extract asset holdings from all consumer types
    aLvl_sim = np.concatenate([ThisType.aLvl for ThisType in MyTypes])

    # Calculate simulated Lorenz curve points
    lorenz_sim = get_lorenz_shares(aLvl_sim, percentiles=[0.2, 0.4, 0.6, 0.8])

    # Calculate the Euclidean distance between the simulated and actual Lorenz curves
    lorenz_distance = np.sqrt(np.sum((lorenz_SCF - lorenz_sim) ** 2))

    # Return the Lorenz distance
    return lorenz_distance


# %% [markdown]
# ## ...and the Marginal Propensity to Consume
#
# Now let's look at the aggregate MPC.  In the code block below, write a function that produces text output of the following form:
#
# $\texttt{The 35th percentile of the MPC is 0.15623}$
#
# Your function should take two inputs: a list of types of consumers and an array of percentiles (numbers between 0 and 1). It should return no outputs, merely print to screen one line of text for each requested percentile.  The model is calibrated at a quarterly frequency, but Carroll et al report MPCs at an annual frequency. To convert, use the formula:
#
# $\kappa_{Y} \approx 1.0 - (1.0 - \kappa_{Q})^4$


# %%
# Write a function to tell us about the distribution of the MPC in this code block, then test it!
# You will almost surely find it useful to use a for loop in this function.
def describeMPCdstn(SomeTypes, percentiles):
    MPC_sim = np.concatenate([ThisType.MPCnow for ThisType in SomeTypes])
    MPCpercentiles_quarterly = get_percentiles(MPC_sim, percentiles=percentiles)
    MPCpercentiles_annual = 1.0 - (1.0 - MPCpercentiles_quarterly) ** 4

    for j in range(len(percentiles)):
        print(
            "The "
            + str(100 * percentiles[j])
            + "th percentile of the MPC is "
            + str(MPCpercentiles_annual[j])
        )


describeMPCdstn(MyTypes, np.linspace(0.05, 0.95, 19))

# %% [markdown]
# # If You Get Here ...
#
# If you have finished the above exercises quickly and have more time to spend on this assignment, for extra credit you can do the same exercise where, instead of exploring the consequences of alternative values of relative risk aversion $\rho$, you should test the consequences of different values of the growth factor $\Gamma$ that lead to the same $\check{m}$.

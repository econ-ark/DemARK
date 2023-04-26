# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,collapsed,code_folding,-autoscroll
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-widgets,-varInspector
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
# # Making Structural Estimates From Empirical Results
#
# [![badge](https://img.shields.io/badge/Launch%20using%20-Econ--ARK-blue)](https://econ-ark.org/materials/structural-estimates-from-empirical-mpcs-fagereng-et-al#launch)
#
# This notebook conducts a quick and dirty structural estimation based on Table 9 of ["MPC Heterogeneity and Household Balance Sheets" by Fagereng, Holm, and Natvik](https://economicdynamics.org/meetpapers/2017/paper_65.pdf) <cite data-cite="6202365/SUE56C4B"></cite>, who use Norweigian administrative data on income, household assets, and lottery winnings to examine the MPC from transitory income shocks (lottery prizes).  Their Table 9 reports an estimated MPC broken down by quartiles of bank deposits and
# prize size; this table is reproduced here as $\texttt{MPC_target_base}$.  In this demo, we use the Table 9 estimates as targets in a simple structural estimation, seeking to minimize the sum of squared differences between simulated and estimated MPCs by changing the (uniform) distribution of discount factors.  The essential question is how well their results be rationalized by a simple one-asset consumption-saving model. (Note that the paper was later published under a different [version](https://www.aeaweb.org/articles?id=10.1257/mac.20190211) which unfortunately excluded table 9.)
#
#
# The function that estimates discount factors includes several options for estimating different specifications:
#
# 1. TypeCount : Integer number of discount factors in discrete distribution; can be set to 1 to turn off _ex ante_ heterogeneity (and to discover that the model has no chance to fit the data well without such heterogeneity).
# 2. AdjFactor : Scaling factor for the target MPCs; user can try to fit estimated MPCs scaled down by (e.g.) 50%.
# 3. T_kill    : Maximum number of years the (perpetually young) agents are allowed to live.  Because this is quick and dirty, it's also the number of periods to simulate.
# 4. Splurge   : Amount of lottery prize that an individual will automatically spend in a moment of excitement (perhaps ancient tradition in Norway requires a big party when you win the lottery), before beginning to behave according to the optimal consumption function.  The patterns in Table 9 can be fit much better when this is set around \$700 --> 0.7.  That doesn't seem like an unreasonable amount of money to spend on a memorable party.
# 5. do_secant : Boolean indicator for whether to use "secant MPC", which is average MPC over the range of the prize.  MNW believes authors' regressions are estimating this rather than point MPC.  When False, structural estimation uses point MPC after receiving prize.  NB: This is incompatible with Splurge > 0.
# 6. drop_corner : Boolean for whether to include target MPC in the top left corner, which is greater than 1.  Authors discuss reasons why the MPC from a transitory shock *could* exceed 1.  Option is included here because this target tends to push the estimate around a bit.

# %% {"code_folding": [0]}
# Import python tools


import numpy as np
from copy import deepcopy

# %% {"code_folding": []}
# Import needed tools from HARK

from HARK.distribution import Uniform
from HARK.utilities import get_percentiles
from HARK.parallel import multi_thread_commands
from HARK.estimation import minimize_nelder_mead
from HARK.ConsumptionSaving.ConsIndShockModel import *


init_infinite = {
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
    "UnempPrbRet": 0.07,
    "IncUnempRet": 0.15,
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

# %% {"code_folding": []}
# Set key problem-specific parameters

TypeCount = 8  # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0  # Factor by which to scale all of MPCs in Table 9
T_kill = 100  # Don't let agents live past this age
Splurge = 0.0  # Consumers automatically spend this amount of any lottery prize
do_secant = True  # If True, calculate MPC by secant, else point MPC
drop_corner = False  # If True, ignore upper left corner when calculating distance

# %% {"code_folding": [0]}
# Set standard HARK parameter values

base_params = deepcopy(init_infinite)
base_params["LivPrb"] = [0.975]
base_params["Rfree"] = 1.04 / base_params["LivPrb"][0]
base_params["PermShkStd"] = [0.1]
base_params["TranShkStd"] = [0.1]
base_params[
    "T_age"
] = T_kill  # Kill off agents if they manage to achieve T_kill working years
base_params["AgentCount"] = 10000
# From Table 1, in thousands of USD
base_params["pLvlInitMean"] = np.log(23.72)
base_params[
    "T_sim"
] = T_kill  # No point simulating past when agents would be killed off

# %% {"code_folding": [0]}
# Define the MPC targets from Fagereng et al Table 9; element i,j is lottery quartile i, deposit quartile j

MPC_target_base = np.array(
    [
        [1.047, 0.745, 0.720, 0.490],
        [0.762, 0.640, 0.559, 0.437],
        [0.663, 0.546, 0.390, 0.386],
        [0.354, 0.325, 0.242, 0.216],
    ]
)
MPC_target = AdjFactor * MPC_target_base

# %% {"code_folding": [0]}
# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages

lottery_size = np.array([1.625, 3.3741, 7.129, 40.0])

# %% {"code_folding": [0]}
# Make several consumer types to be used during estimation

BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    EstTypeList[-1].seed = j

# %% {"code_folding": []}
# Define the objective function


def FagerengObjFunc(center, spread, verbose=False):
    """
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).

    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
    verbose : bool
        When True, print to screen MPC table for these parameters.  When False,
        print (center, spread, distance).

    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    """
    # Give our consumer types the requested discount factor distribution
    beta_set = (
        Uniform(bot=center - spread, top=center + spread)
        .discretize(N=TypeCount)
        .atoms.flatten()
    )
    for j in range(TypeCount):
        EstTypeList[j].DiscFac = beta_set[j]

    # Solve and simulate all consumer types, then gather their wealth levels
    multi_thread_commands(
        EstTypeList, ["solve()", "initialize_sim()", "simulate()", "unpack_cFunc()"]
    )
    WealthNow = np.concatenate([ThisType.state_now["aLvl"] for ThisType in EstTypeList])

    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = get_percentiles(WealthNow, percentiles=[0.25, 0.50, 0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount, dtype=int)
        for n in range(3):
            WealthQ[ThisType.state_now["aLvl"] > quartile_cuts[n]] += 1
        ThisType.WealthQ = WealthQ

    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
        [[], [], [], []],
    ]

    # Calculate the MPC for each of the four lottery sizes for all agents
    for ThisType in EstTypeList:
        ThisType.simulate(1)
        c_base = ThisType.controls["cNrm"]
        MPC_this_type = np.zeros((ThisType.AgentCount, 4))
        for k in range(4):  # Get MPC for all agents of this type
            Llvl = lottery_size[k]
            Lnrm = Llvl / ThisType.state_now["pLvl"]
            if do_secant:
                SplurgeNrm = Splurge / ThisType.state_now["pLvl"]
                mAdj = ThisType.state_now["mNrm"] + Lnrm - SplurgeNrm
                cAdj = ThisType.cFunc[0](mAdj) + SplurgeNrm
                MPC_this_type[:, k] = (cAdj - c_base) / Lnrm
            else:
                mAdj = ThisType.state_now["mNrm"] + Lnrm
                MPC_this_type[:, k] = cAdj = ThisType.cFunc[0].derivative(mAdj)

        # Sort the MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_this_type[these, k])

    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4, 4))
    for k in range(4):
        for q in range(4):
            MPC_array = np.concatenate(MPC_set_list[k][q])
            simulated_MPC_means[k, q] = np.mean(MPC_array)

    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    diff = simulated_MPC_means - MPC_target
    if drop_corner:
        diff[0, 0] = 0.0
    distance = np.sqrt(np.sum((diff) ** 2))
    if verbose:
        print(simulated_MPC_means)
    else:
        print(center, spread, distance)
    return distance


# %% {"code_folding": []}
# Conduct the estimation

guess = [0.92, 0.03]


def f_temp(x):
    return FagerengObjFunc(x[0], x[1])


opt_params = minimize_nelder_mead(f_temp, guess, verbose=False)
print(
    "Finished estimating for scaling factor of "
    + str(AdjFactor)
    + ' and "splurge amount" of $'
    + str(1000 * Splurge)
)
print("Optimal (beta,nabla) is " + str(opt_params) + ", simulated MPCs are:")
dist = FagerengObjFunc(opt_params[0], opt_params[1], True)
print("Distance from Fagereng et al Table 9 is " + str(dist))

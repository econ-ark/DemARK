# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,collapsed,jupyter,tags,title,-autoscroll
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-widgets,-varInspector
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
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
#     version: 3.9.13
# ---

# %% [markdown]
# # Expectated vs Realized Income Growth in A Standard Life Cycle Model

# %% [markdown]
# This notebook uses the income process in [Cocco, Gomes & Maenhout (2005)](https://academic.oup.com/rfs/article/18/2/491/1599892?login=true) to demonstrate that estimates of a regression of expected income changes on realized income changes are sensitive to the size of transitory shocks.
#
# We first load some tools from the [HARK toolkit](https://github.com/econ-ark/HARK).

# %% jupyter={"source_hidden": true}
import statsmodels.api as sm
from linearmodels.panel.model import PanelOLS
from HARK.distribution import calc_expectation
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)

from HARK.Calibration.Income.IncomeTools import (
    parse_income_spec,
    parse_time_params,
    CGM_income,
)

from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
import pandas as pd
from copy import copy

# %% [markdown]
# We now create a population of agents with the income process of [Cocco, Gomes & Maenhout (2005)](https://academic.oup.com/rfs/article/18/2/491/1599892?login=true), which is implemented as a default calibration in the toolkit.

# %% Alter calibration jupyter={"source_hidden": true}
birth_age = 21
death_age = 66
adjust_infl_to = 1992
income_calib = CGM_income
education = "HS"

# Income specification
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib[education],
    SabelhausSong=True
)

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(
    female=True, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
params = copy(init_lifecycle)
params.update(time_params)
# params.update(dist_params)
params.update(income_params)
params.update(
    {
        "LivPrb": liv_prb,
        "pLvlInitStd": 0.0,
        "PermGroFacAgg": 1.0,
        "UnempPrb": 0.0,
        "UnempPrbRet": 0.0,
        "track_vars": ["pLvl", "t_age", "PermShk", "TranShk"],
        "AgentCount": 200,
        "T_sim": 500,
    }
)

Agent = IndShockConsumerType(**params)
Agent.solve()

# %% Create and solve agent [markdown]
# We simulate a population of agents

# %% Simulation
# %%capture
# Run the simulations
Agent.initialize_sim()
Agent.simulate()

# %% [markdown]
#
# $\newcommand{\Ex}{\mathbb{E}}$
# $\newcommand{\PermShk}{\psi}$
# $\newcommand{\pLvl}{\mathbf{p}}$
# $\newcommand{\pLvl}{P}$
# $\newcommand{\yLvl}{\mathbf{y}}$
# $\newcommand{\yLvl}{Y}$
# $\newcommand{\PermGroFac}{\Gamma}$
# $\newcommand{\UnempPrb}{\wp}$
# $\newcommand{\TranShk}{\theta}$
#
# We assume a standard income process with transitory and permanent shocks:  The consumer's Permanent noncapital income $\pLvl$ grows by a predictable factor $\PermGroFac$ and is subject to an unpredictable multiplicative shock $\Ex_{t}[\PermShk_{t+1}]=1$,
#
# \begin{eqnarray}
# \pLvl_{t+1} & = & \pLvl_{t} \PermGroFac_{t+1} \PermShk_{t+1}, \notag
# \end{eqnarray}
# and, if the consumer is employed, actual income $Y$ is permanent income multiplied by a transitory shock $\Ex_{t}[\TranShk_{t+1}]=1$,
# \begin{eqnarray}
# \yLvl_{t+1} & = & \pLvl_{t+1} \TranShk_{t+1}, \notag
# \end{eqnarray}
#
# <!--- There is also a probability $\UnempPrb$ that the consumer will be temporarily unemployed and experience income of $\TranShk^{\large u}  = 0$.  We construct $\TranShk^{\large e}$ so that its mean value is $1/(1-\UnempPrb)$ because in that case the mean level of the transitory shock (accounting for both unemployed and employed states) is exactly
#
# \begin{eqnarray}
# \Ex_{t}[\TranShk_{t+1}] & = & \TranShk^{\large{u}}  \times \UnempPrb + (1-\UnempPrb) \times \Ex_{t}[\TranShk^{\large{e}}_{t+1}] \notag
# \\ & = & 0 \times \UnempPrb + (1-\UnempPrb) \times 1/(1-\UnempPrb)  \notag
# \\ & = & 1. \notag
# \end{eqnarray}
# --->
#
# $\Gamma_{t}$ captures the predictable life cycle profile of income growth (faster when young, slower when old).  See [our replication of CGM-2005](https://github.com/econ-ark/CGMPortfolio/blob/master/Code/Python/CGMPortfolio.ipynb) for a detailed account of how these objects map to CGM's notation.

# %% [markdown]
# Now define $\newcommand{\yLog}{y}\newcommand{\pLog}{p}\yLog = \log \yLvl,\pLog=\log \pLvl$ and similarly for other variables.
#
# Using this notation, we construct all the necessary inputs to the regressors. The main input is the expected income growth of every agent at every time period, which is given by
# \begin{equation}
# \begin{split}
# \Ex_t[\yLvl_{t+1}/\yLvl_{t}] &= \mathbb{E}_t[\left(\frac{\theta_{t+1}\pLvl_{t} \PermGroFac_{t+1} \PermShk_{t+1}}{\theta_{t}P_{t}}\right)]\\
#  &= \left(\frac{\PermGroFac_{t+1}}{\theta_{t}}\right)\\
# \Ex_t[\yLog_{t+1} - \yLog_{t}] & = \log \Gamma_{t+1}-\log \theta_t
# \end{split}
# \end{equation}
#

# %% Compute expectations jupyter={"source_hidden": true}
exp = [
    calc_expectation(Agent.IncShkDstn[i], func=lambda x: x[0] * x[1])
    for i in range(Agent.T_cycle)
]
exp_df = pd.DataFrame(
    {
        "exp_prod": exp,
        "PermGroFac": Agent.PermGroFac,
        "Age": [x + birth_age for x in range(Agent.T_cycle)],
    }
)

raw_data = {
    "Age": Agent.history["t_age"].T.flatten() + birth_age - 1,
    "pLvl": Agent.history["pLvl"].T.flatten(),
    "PermShk": Agent.history["PermShk"].T.flatten(),
    "TranShk": Agent.history["TranShk"].T.flatten(),
}

Data = pd.DataFrame(raw_data)

# Create an individual id
Data["id"] = (Data["Age"].diff(1) < 0).cumsum()

Data["Y"] = Data.pLvl * Data.TranShk

# Find Et[Yt+1 - Yt]
Data = Data.join(exp_df.set_index("Age"), on="Age", how="left")
Data["ExpIncChange"] = Data["pLvl"] * (
    Data["PermGroFac"] * Data["exp_prod"] - Data["TranShk"]
)

Data["Y_change"] = Data.groupby("id")["Y"].diff(1)
# %% [markdown]
# A corresponding version of this relationship can be estimated in simulated data:
# \begin{equation*}
#         \Ex_t[\Delta y_{i,t+1}] = \gamma_{0} + \gamma_{1} \Delta y_{i,t} + f_i + \epsilon_{i,t}
# \end{equation*}
# We now estimate an analogous regression in our simulated population.

# %% jupyter={"source_hidden": true}
Data = Data.set_index(["id", "Age"])

# Create the variables they actually use
Data["ExpBin"] = 0
Data.loc[Data["ExpIncChange"] > 0, "ExpBin"] = 1
Data.loc[Data["ExpIncChange"] < 0, "ExpBin"] = -1

Data["ChangeBin"] = 0
Data.loc[Data["Y_change"] > 0, "ChangeBin"] = 1
Data.loc[Data["Y_change"] < 0, "ChangeBin"] = -1

mod = PanelOLS(Data.ExpBin, sm.add_constant(Data.ChangeBin), entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# %% [markdown]
# The estimated $\hat{\gamma}_{1}$ is negative because in usual life-cycle calibrations, transitory shocks are volatile enough that mean reversion of transitory fluctuations is a stronger force than persistent trends in income age-profiles.
#
# However, with less volatile transitory shocks, the regression coefficient would be positive. We demonstrate this by shutting off transitory shocks, simulating another population of agents, and re-running the regression.

# %%
# %%capture
params_no_transitory = copy(params)
params_no_transitory.update({"TranShkStd": [0.0] * len(params["TranShkStd"])})

# Create agent
Agent_nt = IndShockConsumerType(**params_no_transitory)
Agent_nt.solve()
# Run the simulations
Agent_nt.initialize_sim()
Agent_nt.simulate()


# %% jupyter={"source_hidden": true}
exp = [
    calc_expectation(Agent_nt.IncShkDstn[i], func=lambda x: x[0] * x[1])
    for i in range(Agent_nt.T_cycle)
]
exp_df = pd.DataFrame(
    {
        "exp_prod": exp,
        "PermGroFac": Agent_nt.PermGroFac,
        "Age": [x + birth_age for x in range(Agent.T_cycle)],
    }
)

raw_data = {
    "Age": Agent_nt.history["t_age"].T.flatten() + birth_age - 1,
    "pLvl": Agent_nt.history["pLvl"].T.flatten(),
    "PermShk": Agent_nt.history["PermShk"].T.flatten(),
    "TranShk": Agent_nt.history["TranShk"].T.flatten(),
}

Data = pd.DataFrame(raw_data)

# Create an individual id
Data["id"] = (Data["Age"].diff(1) < 0).cumsum()

Data["Y"] = Data.pLvl * Data.TranShk

# Find Et[Yt+1 - Yt]
Data = Data.join(exp_df.set_index("Age"), on="Age", how="left")
Data["ExpIncChange"] = Data["pLvl"] * (
    Data["PermGroFac"] * Data["exp_prod"] - Data["TranShk"]
)

Data["Y_change"] = Data.groupby("id")["Y"].diff(1)

# %% jupyter={"source_hidden": true}
# Create variables
Data["ExpBin"] = 0
Data.loc[Data["ExpIncChange"] > 0, "ExpBin"] = 1
Data.loc[Data["ExpIncChange"] < 0, "ExpBin"] = -1

Data["ChangeBin"] = 0
Data.loc[Data["Y_change"] > 0, "ChangeBin"] = 1
Data.loc[Data["Y_change"] < 0, "ChangeBin"] = -1

Data = Data.set_index(["id", "Age"])
mod = PanelOLS(Data.ExpBin, sm.add_constant(Data.ChangeBin), entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# %% [markdown]
# The estimated $\hat{\gamma}_{1}$ when there are no transitory shocks is positive.

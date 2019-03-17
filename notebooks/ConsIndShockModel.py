# ---
# jupyter:
#   '@webio':
#     lastCommId: 779f6c5616b04b58baaaa0f6c348270c
#     lastKernelId: a944b08f-0ae0-4c26-883f-9fab53a82ac3
#   jupytext:
#     formats: ipynb,py:percent
#     metadata_filter:
#       cells: collapsed
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
#     version: 3.6.7
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

# %% [markdown]
# # ConsIndShockModel: Consumption With Shocks

# %% {"code_folding": []}
# Initial imports and notebook setup, click arrow to show
import sys
import os

from HARK.ConsumptionSaving.ConsIndShockModel import *
import HARK.ConsumptionSaving.ConsumerParameters as Params
from HARK.utilities import plotFuncsDer, plotFuncs
from time import clock
mystr = lambda number : "{:.4f}".format(number)

# %% [markdown]
# Defines classes to solve canonical consumption-saving models with idiosyncratic shocks to income.  All models here assume CRRA utility with geometric discounting, no bequest motive, and income shocks are fully transitory or fully permanent.
#
# ConsIndShockModel currently solves three types of models:
# 1. A basic "perfect foresight" consumption-saving model with no uncertainty.
# 2. A consumption-saving model with risk over transitory and permanent income shocks.
# 3. The model described in (2), with an interest rate for debt that differs from the interest rate for savings.
#
# See [NARK](https://github.com/econ-ark/NARK) for information on variable naming conventions.
# See [HARK documentation](https://github.com/econ-ark/HARK/Documentation) for brief mathematical descriptions of the models being solved.  Detailed mathematical references are referenced _in situ_ below.

# %% [markdown]
# ## Perfect Foresight CRRA Utility Consumer
#
# The $\mathtt{PerfForesightConsumerType}$ class defines the solution for the problem of a consumer with Constant Relative Risk Aversion utility 
# $$\newcommand{\CRRA}{\rho}$$
# \begin{equation}
# U(C) = \frac{C^{1-\CRRA}}{1-\rho}
# \end{equation}
# has perfect foresight about everything except the (stochastic) date of death, which occurs with constant probability implying a "survival probability" $\newcommand{\LivPrb}{\aleph}\LivPrb < 1$.  Permanent labor income $P_t$ grows from period to period by a factor $\Gamma_t$.  At the beginning of each period $t$, the consumer has some amount of market resources $M_t$ (which includes both market wealth and currrent income) and must choose how much of those resources to consume $C_t$ and how much to retain in a riskless asset $A_t$ which will earn return factor $R$. The agent's flow of utility $U(C_t)$ from consumption is geometrically discounted by factor $\beta$. Between periods, the agent survives with probability $\newcommand{\LivFac}{\aleph}{\LivFac_{t}}$ which results in a further downweighting of future utility because the consumer does not receive utility after death.  For notational simplicity, we omit $\LivFac$ from the statement of the problem (effectively assuming $\LivFac=1$). 
#
# The agent's problem can be written in Bellman form as:
# $$\newcommand{\DiscFac}{\beta}\renewcommand{\LivFac}{\aleph}{}$$
# \begin{eqnarray*}
# V_t(M_t,P_t) &=& \max_{C_t}~U(C_t) ~+ \phantom{\LivFac} \DiscFac V_{t+1}(M_{t+1},P_{t+1}), \\
# & s.t. & \\
# %A_t &=& M_t - C_t, \\
# M_{t+1} &=& R (M_{t}-C_{t}) + Y_{t+1}, \\
# P_{t+1} &=& \Gamma_{t+1} P_t, \\
# \end{eqnarray*}
#
# A particular perfect foresight agent's problem can be characterized by values of risk aversion $\rho$, discount factor $\beta$, and return factor $R$, along with sequences of income growth factors $\{ \Gamma_t \}$ and survival probabilities $\{\LivPrb_t = \LivPrb = 1\}$ (which are allowed to vary by age but which for present purposes we will assume are time invariant at $\LivPrb$.  To keep things simple, let's forget about "sequences" of income growth and mortality, and just think about an $\textit{infinite horizon}$ consumer with constant income growth and survival probability of $\LivFac=1$.
#
#
# Solve the model described above and in [PerfForesightCRRA](http://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA)

# %%
PFexample = PerfForesightConsumerType(**Params.init_perfect_foresight)
PFexample.cycles = 0 # Make this type have an infinite horizon

PFexample.solve()
PFexample.unpackcFunc()

# Plot the perfect foresight consumption function
print('Linear consumption function:')
mMin = PFexample.solution[0].mNrmMin
plotFuncs(PFexample.cFunc[0],mMin,mMin+10)

PFexample.timeFwd()
PFexample.T_sim = 120 # Set number of simulation periods
PFexample.track_vars = ['mNrmNow']
PFexample.initializeSim()
PFexample.simulate()

# %% [markdown]
# ## Consumer with idiosyncratic income shocks
#
# Solve a model like the one analyzed in [BufferStockTheory](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/)
#
# Specifically, our new type of consumer receives two income shocks at the beginning of each period: a completely transitory shock $\newcommand{\tShkEmp}{\theta}{\tShkEmp_t}$ and a completely permanent shock $\newcommand{\pShk}{\psi}{\pShk_t}$.  Moreover, lenders will not let the agent borrow money such that his ratio of end-of-period assets $A_t$ to permanent income $P_t$ is less than $\underline{a}$.  As with the perfect foresight problem, this model can be framed in terms of $\textit{normalized}$ variables, e.g. $m_t \equiv M_t/P_t$.  (See [here](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/) for all the theory).
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t} {~} U(c_t) + \phantom{\LivPrb} \beta  \mathbb{E}_{t} [(\Gamma_{t+1}\psi_{t+1})^{1-\rho} v_{t+1}(m_{t+1}) ], \\
# a_t &=& m_t - c_t, \\
# a_t &\geq& \underline{a}, \\
# m_{t+1} &=& R/(\Gamma_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
# \mathbb{E}[\psi]=\mathbb{E}[\theta] &=& 1, \\
# u(c) &=& \frac{c^{1-\rho}}{1-\rho}.
# \end{eqnarray*}
#
# HARK represents agents with this kind of problem as instances of the class $\texttt{IndShockConsumerType}$.  To create an $\texttt{IndShockConsumerType}$, we must specify the same set of parameters as for a $\texttt{PerfForesightConsumerType}$, as well as an artificial borrowing constraint $\underline{a}$ and a stochastic process for the income shocks. 
#
# The user can specify any desired discrete approximation to a continuous distribution for the IID shocks.  We have built-in tools for constructing thes as discrete equiprobably approximations to lognormal, which is the default assumption.

# %%
IndShockExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
IndShockExample.cycles = 0 # Make this type have an infinite horizon

start_time = clock()
IndShockExample.solve()
end_time = clock()
print('Solving a consumer with idiosyncratic shocks took ' + mystr(end_time-start_time) + ' seconds.')
IndShockExample.unpackcFunc()
IndShockExample.timeFwd()

# Plot the consumption function and MPC for the infinite horizon consumer
print('Concave consumption function:')
plotFuncs(IndShockExample.cFunc[0],IndShockExample.solution[0].mNrmMin,5)
print('Marginal propensity to consume function:')
plotFuncsDer(IndShockExample.cFunc[0],IndShockExample.solution[0].mNrmMin,5)

# Compare the consumption functions for the perfect foresight and idiosyncratic
# shock types.  Risky income cFunc asymptotically approaches perfect foresight cFunc.
print('Consumption functions for perfect foresight vs idiosyncratic shocks:')
plotFuncs([PFexample.cFunc[0],IndShockExample.cFunc[0]],IndShockExample.solution[0].mNrmMin,100)

# Compare the value functions for the two types
if IndShockExample.vFuncBool:
    print('Value functions for perfect foresight vs idiosyncratic shocks:')
    plotFuncs([PFexample.solution[0].vFunc,IndShockExample.solution[0].vFunc],
                  IndShockExample.solution[0].mNrmMin+0.5,10)

# Simulate some data; results stored in mNrmNow_hist, cNrmNow_hist, and pLvlNow_hist
IndShockExample.T_sim = 120
IndShockExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow']
IndShockExample.makeShockHistory() # This is optional, simulation will draw shocks on the fly if it isn't run.
IndShockExample.initializeSim()
IndShockExample.simulate()


# %% [markdown]
# ## Idiosyncratic shocks consumer with a finite lifecycle
#
# Models of this kinds are described in [SolvingMicroDSOPs](http://econ.jhu.edu/people/ccarroll/SolvingMicroDSOPs) and an example is solved in the [SolvingMicroDSOPs REMARK](https://github.com/econ-ark/REMARK/REMARKs/SolvingMicroDSOPs.md).

# %%
LifecycleExample = IndShockConsumerType(**Params.init_lifecycle)
LifecycleExample.cycles = 1 # Make this consumer live a sequence of periods -- a lifetime -- exactly once

start_time = clock()
LifecycleExample.solve()
end_time = clock()
print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
LifecycleExample.unpackcFunc()
LifecycleExample.timeFwd()

# Plot the consumption functions during working life
print('Consumption functions while working:')
mMin = min([LifecycleExample.solution[t].mNrmMin for t in range(LifecycleExample.T_cycle)])
plotFuncs(LifecycleExample.cFunc[:LifecycleExample.T_retire],mMin,5)

# Plot the consumption functions during retirement
print('Consumption functions while retired:')
plotFuncs(LifecycleExample.cFunc[LifecycleExample.T_retire:],0,5)
LifecycleExample.timeRev()

# Simulate some data; results stored in mNrmNow_hist, cNrmNow_hist, pLvlNow_hist, and t_age_hist
LifecycleExample.T_sim = 120
LifecycleExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow','t_age']
LifecycleExample.initializeSim()
LifecycleExample.simulate()

# %% [markdown]
# ## "Cyclical" consumer type
# Make and solve a "cyclical" consumer type who lives the same four quarters repeatedly.
# The consumer has income that greatly fluctuates throughout the year.

# %%
CyclicalExample = IndShockConsumerType(**Params.init_cyclical)
CyclicalExample.cycles = 0

start_time = clock()
CyclicalExample.solve()
end_time = clock()
print('Solving a cyclical consumer took ' + mystr(end_time-start_time) + ' seconds.')
CyclicalExample.unpackcFunc()
CyclicalExample.timeFwd()

# Plot the consumption functions for the cyclical consumer type
print('Quarterly consumption functions:')
mMin = min([X.mNrmMin for X in CyclicalExample.solution])
plotFuncs(CyclicalExample.cFunc,mMin,5)

# Simulate some data; results stored in cHist, mHist, bHist, aHist, MPChist, and pHist
CyclicalExample.T_sim = 480
CyclicalExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow','t_cycle']
CyclicalExample.initializeSim()
CyclicalExample.simulate()

# %% [markdown]
# ## Agent with a kinky interest rate (Rboro > RSave)
#
# Models of this kind are analyzed in [A Theory of the Consumption Function, With
# and Without Liquidity Constraints](http://www.econ2.jhu.edu/people/ccarroll/ATheoryv3JEP.pdf)
# and the [expanded edition](http://www.econ2.jhu.edu/people/ccarroll/ATheoryv3NBER.pdf).

# %%
KinkyExample = KinkedRconsumerType(**Params.init_kinked_R)
KinkyExample.cycles = 0 # Make the Example infinite horizon

start_time = clock()
KinkyExample.solve()
end_time = clock()
print('Solving a kinky consumer took ' + mystr(end_time-start_time) + ' seconds.')
KinkyExample.unpackcFunc()
print('Kinky consumption function:')
KinkyExample.timeFwd()
plotFuncs(KinkyExample.cFunc[0],KinkyExample.solution[0].mNrmMin,5)

KinkyExample.T_sim = 120
KinkyExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow']
KinkyExample.initializeSim()
KinkyExample.simulate()

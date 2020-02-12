# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # A Gentle Introduction to Buffer Stock Saving 
#
# This notebook explores the behavior of a consumer identical to the perfect foresight consumer described in [Gentle-Intro-To-HARK-PerfForesightCRRA](https://econ-ark.org/materials/Gentle-Intro-To-HARK-PerfForesightCRRA) except that now the model incorporates income uncertainty.
#
# Specifically, our new type of consumer receives two income shocks at the beginning of each period: a completely transitory shock $\theta_t$ and a completely permanent shock $\psi_t$.  Moreover, lenders set a limit on borrowing: The ratio of end-of-period assets $A_t$ to permanent income $P_t$ must be less greater than $\underline{a} \leq 0$. As with the perfect foresight problem, this model can be framed in terms of _normalized_ variables, e.g. $m_t \equiv M_t/P_t$.  (See [here](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/) for all the theory).
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t} U(c_t) + \beta (1 - \mathsf{D}_{t+1}) \mathbb{E} [(\Gamma_{t+1}\psi_{t+1})^{1-\rho} v_{t+1}(m_{t+1}) ], \\
# a_t &=& m_t - c_t, \\
# a_t &\geq& \underline{a}, \\
# m_{t+1} &=& R/(\Gamma_{t+1} \psi_{t+1}) a_t + \theta_{t+1}, \\
# (\psi_t,\theta_t) \sim F_{t}, &\qquad& \mathbb{E} [F_{\psi t}] = [F_{\theta t}] = 1, \\
# U(c) &=& \frac{c^{1-\rho}}{1-\rho}.
# \end{eqnarray*}
#
# HARK represents agents with this kind of problem as instances of the class $\texttt{IndShockConsumerType}$.  To create a $\texttt{IndShockConsumerType}$ instance, we must specify the same set of parameters as for a $\texttt{PerfForesightConsumerType}$, as well as an artificial borrowing constraint $\underline{a}$ and a sequence of income shock distributions $\{F_t\}$. It's easy enough to pick a borrowing constraint-- say, $\underline{a} = 0$ so that the consumer cannot borrow at all.
#
# Computers are discrete devices; even if somehow we knew with certainty that the distributions of the transitory and permanent shocks were, say, continuously lognormally distributed, in order to be represented on a computer those distributions would need to be approximated by a finite and discrete set of points.  A large literature in numerical computation explores ways to construct such approximations; probably the easiest example to understand is the equiprobable approximation, in which the continuous distribution is represented by a set of $N$ outcomes that are equally likely to occur. 
#
# In the case of a single variable (say, the permanent shock $\psi$), and when the number of equiprobable points is, say, 5, the procedure is to construct a list: $psi_{0}$ is the mean value of the continuous $\psi$ given that the draw of $\psi$ is in the bottom 20 percent of the distribution of the continuous $\psi$.  $\\psi_{1}$ is the mean value of $\psi$ given that the draw is between the 20th and 40th percentiles, and so on.  The expectation of some expression $f(\psi)$ can be very quickly calculated by:
#
# $$ 
# \mathbb{E}_{t}[f(\psi)] \approx (1/N) \sum_{i=0}^{N-1} f(\psi_{i})
# $$

# %% [markdown]
# In principle, any smooth multivariate continuous distribution can be approximated to an arbitrary degree of accuracy with a fine enough matrix of points and their corresponding probailities.  This is, in fact, the fundamental way that HARK represents uncertainty: By a specifying a multidimensional array containing joint probabilities of the realizations of the shocks.
#
# The simplest assumption (and therefore the default choice in $\texttt{IndShockConsumerType}$) is that the transitory and permanent shocks are independent.  The permanent shock is assumed to be lognormal, while the transitory shock has two components: A probability $\wp$ that the consumer is unemployed, in which case $\theta=\underline{\theta}$, and a probability $(1-\wp)$ of a shock that is a lognormal with a mean chosen so that $\mathbb{E}_{t}[\theta_{t+n}]=1$.
#
# The $\texttt{IndShockConsumerType}$ inherits all of the parameters of the original $\texttt{PerfForesightConsumerType}$ class.  Given the assumptions above, we need to specify the extra parameters to specify the income shock distribution and the artificial borrowing constraint. As before, we'll make a dictionary:
#
#
# | Param | Description | Code | Value |
# | :---: | ---         | ---  | :---: |
# | $\underline{a}$ | Artificial borrowing constraint | $\texttt{BoroCnstArt}$ | 0.0 |
# | $\sigma_\psi$ | Underlying stdev of permanent income shocks | $\texttt{PermShkStd}$ | 0.1 |
# | $\sigma_\theta^{e}$ | Underlying stdev of transitory income shocks | $\texttt{TranShkStd}$ | 0.1 |
# | $N_\psi$ | Number of discrete permanent income shocks | $\texttt{PermShkCount}$ | 7 |
# | $N_\theta$ | Number of discrete transitory income shocks | $\texttt{TranShkCount}$ | 7 |
# | $\mho$ | Unemployment probability | $\texttt{UnempPrb}$ | 0.05 |
# | $\underline{\theta}$ | Transitory shock when unemployed | $\texttt{IncUnemp}$ | 0.3 |

# %% {"code_folding": []}
# This cell has just a bit of initial setup. You can click the triangle to the left to expand it.
# Click the "Run" button immediately above the notebook in order to execute the contents of any cell
# WARNING: Each cell in the notebook relies upon results generated by previous cells
#   The most common problem beginners have is to execute a cell before all its predecessors
#   If you do this, you can restart the kernel (see the "Kernel" menu above) and start over
# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import HARK 
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.4f}".format(number)
from HARK.utilities import plotFuncs

# %% {"code_folding": [0, 2]}
# This cell defines a parameter dictionary for making an instance of IndShockConsumerType.

IndShockDictionary = {
    'CRRA': 2.5,          # The dictionary includes our original parameters...
    'Rfree': 1.03,
    'DiscFac': 0.96,
    'LivPrb': [0.98],
    'PermGroFac': [1.01],
    'PermShkStd': [0.1],  # ... and the new parameters for constructing the income process.    
    'PermShkCount': 7,
    'TranShkStd': [0.1],
    'TranShkCount': 7,
    'UnempPrb': 0.05,
    'IncUnemp': 0.3,
    'BoroCnstArt': 0.0,
    'aXtraMin': 0.001,    # aXtra parameters specify how to construct the grid of assets.
    'aXtraMax': 50.,      # Don't worry about these for now
    'aXtraNestFac': 3,
    'aXtraCount': 48,
    'aXtraExtra': [None],
    'vFuncBool': False,   # These booleans indicate whether the value function should be calculated
    'CubicBool': False,   # and whether to use cubic spline interpolation. You can ignore them.
    'aNrmInitMean' : -10.,
    'aNrmInitStd' : 0.0,  # These parameters specify the (log) distribution of normalized assets
    'pLvlInitMean' : 0.0, # and permanent income for agents at "birth". They are only relevant in
    'pLvlInitStd' : 0.0,  # simulation and you don't need to worry about them.
    'PermGroFacAgg' : 1.0,
    'T_retire': 0,        # What's this about retirement? ConsIndShock is set up to be able to
    'UnempPrbRet': 0.0,   # handle lifecycle models as well as infinite horizon problems. Swapping
    'IncUnempRet': 0.0,   # out the structure of the income process is easy, but ignore for now.
    'T_age' : None,
    'T_cycle' : 1,
    'cycles' : 0,
    'AgentCount': 10000,
    'tax_rate':0.0,
}
        
# Hey, there's a lot of parameters we didn't tell you about!  Yes, but you don't need to
# think about them for now.

# %% [markdown]
# As before, we need to import the relevant subclass of $\texttt{AgentType}$ into our workspace, then create an instance by passing the dictionary to the class as if the class were a function.

# %%
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
IndShockExample = IndShockConsumerType(**IndShockDictionary)

# %% [markdown]
# Now we can solve our new agent's problem just like before, using the $\texttt{solve}$ method.

# %%
IndShockExample.solve()
plotFuncs(IndShockExample.solution[0].cFunc,0.,10.)

# %% [markdown]
# ## Changing Constructed Attributes
#
# In the parameter dictionary above, we chose values for HARK to use when constructing its numeric representation of $F_t$, the joint distribution of permanent and transitory income shocks. When $\texttt{IndShockExample}$ was created, those parameters ($\texttt{TranShkStd}$, etc) were used by the **constructor** or **initialization** method of $\texttt{IndShockConsumerType}$ to construct an attribute called $\texttt{IncomeDstn}$.
#
# Suppose you were interested in changing (say) the amount of permanent income risk.  From the section above, you might think that you could simply change the attribute $\texttt{TranShkStd}$, solve the model again, and it would work.
#
# That's _almost_ true-- there's one extra step. $\texttt{TranShkStd}$ is a primitive input, but it's not the thing you _actually_ want to change. Changing $\texttt{TranShkStd}$ doesn't actually update the income distribution... unless you tell it to (just like changing an agent's preferences does not change the consumption function that was stored for the old set of parameters -- until you invoke the $\texttt{solve}$ method again).  In the cell below, we invoke the method $\texttt{updateIncomeProcess}$ so HARK knows to reconstruct the attribute $\texttt{IncomeDstn}$.

# %%
OtherExample = deepcopy(IndShockExample)  # Make a copy so we can compare consumption functions
OtherExample.PermShkStd = [0.2]           # Double permanent income risk (note that it's a one element list)
OtherExample.updateIncomeProcess()        # Call the method to reconstruct the representation of F_t
OtherExample.solve()

# %% [markdown]
# In the cell below, use your blossoming HARK skills to plot the consumption function for $\texttt{IndShockExample}$ and $\texttt{OtherExample}$ on the same figure.

# %% {"code_folding": []}
# Use the line(s) below to plot the consumptions functions against each other


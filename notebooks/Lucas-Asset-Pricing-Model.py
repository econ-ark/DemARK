# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lucas Asset Pricing Model
#
# ## A notebook by [Christopher D. Carroll](http://www.econ2.jhu.edu/people/ccarroll/) and [Mateo Velásquez-Giraldo](https://mv77.github.io/)
# ### Inspired by its [Quantecon counterpart](https://julia.quantecon.org/multi_agent_models/lucas_model.html)
#
# This notebook presents simple computational tools to solve Lucas' asset-pricing model when the logarithm of the asset's dividend follows an autoregressive process of order 1,
#
# \begin{equation*}
# \ln d_{t+1} = \alpha \ln d_t + \varepsilon_{t+1}.
# \end{equation*}
#
# A presentation of this model can be found in [Christopher D. Carroll's lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/AssetPricing/LucasAssetPrice/). 
#
# Those notes [derive](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/AssetPricing/LucasAssetPrice/#pofc) from the Bellman equation a relationship between the price of the asset in the current period $t$ and the next period $t+1$:  
#
# \begin{equation*}
# P_{t} = 
# \overbrace{\left(\frac{1}{1+\vartheta}\right)}
# ^{\beta}\mathbb{E}_{t}\left[ \frac{u^{\prime}(d_{t+1})}{u^{\prime}(d_t)} (P_{t+1} + d_{t+1}) \right]
# \end{equation*}
#
# The equilibrium pricing equation is a relationship between prices and dividend (a "pricing kernel") $P^{*}(d)$ such that, if everyone _believes_ that to be the pricing kernel, everyone's Euler equation will be satisfied:
#
# \begin{equation*}
# P^*(d_t) = \left(\frac{1}{1+\vartheta}\right)\mathbb{E}_{t}\left[ \frac{u^{\prime}(d_{t+1})}{u^{\prime}(d_t)} (P^*(d_{t+1}) + d_{t+1}) \right]
# \end{equation*}
#
# As noted in the handout, there are some special circumstances in which it is possible to solve for $P^{*}$ analytically:
#
# | Shock Process | CRRA | Solution for Pricing Kernel | 
# | --- | --- | --- |
# | bounded | 1 (log) | $P^*(d) = \frac{d}{\vartheta}$ |
# | lognormal, mean 1 | $\rho$ | $P^*(d) = d_t^\rho\ e^{\rho(\rho-1)\sigma^2/2}\frac{\beta}{1-\beta}$ |
#
# However, under less special circumstances, the only way to obtain the pricing function $P^{*}$ is by solving for it numerically, as outlined below.

# %% [markdown]
# # Finding the equilibrium pricing function.
#
# We know that the equilibrium pricing function must satisfy the equation above. Let's define an operator that allows us to evaluate whether any candidate pricing function satisfies this requirement.
#
# Let $T$ be an operator which takes as argument a function and returns another function (these are usually called [functionals or higher-order functions](https://en.wikipedia.org/wiki/Functional_(mathematics))). For some function $f$, denote with $T[f]$ the function that results from applying $T$ to $f$. Then, for any real number $x$, $T[f](x)$ will be the real number that one obtains when the function $T[f]$ is given $x$ as an input.
#
# We define our particular operator as follows. For any function $g:\mathbb{R}\rightarrow\mathbb{R}$, $T[g]$ is obtained as
#
# \begin{equation*}
# \forall d_t \in \mathbb{R},\,\,\,\, T[g](d_t) := \beta\mathbb{E}_{t}\left[ \frac{u^{\prime}(d_{t+1})}{u^{\prime}(d_t)} (f(d_{t+1}) + d_{t+1}) \right].
# \end{equation*}
#
#
# We can use $T$ to re-express our pricing equation. If $P^*(\cdot)$ is our equilibrium pricing funtion, it must satisfy
#
# \begin{equation*}
# \forall d_t,\,\,\,\,P^*(d_t) = \beta\mathbb{E}_{t}\left[ \frac{u^{\prime}(d_{t+1})}{u^{\prime}(d_t)} (P^*(d_{t+1}) + d_{t+1}) \right] = T[P^*](d_t).
# \end{equation*}
# or, expressed differently,
# \begin{equation*}
# P^* = T[P^*].
# \end{equation*}
#
# Our equilibrium pricing function is therefore a *fixed point* of the operator $T$.
#
# It turns out that $T$ is a [contraction mapping](https://en.wikipedia.org/wiki/Contraction_mapping). This is useful because it implies, through [Banach's fixed-point theorem](https://en.wikipedia.org/wiki/Contraction_mapping), that:
# - $T$ has **exactly one** fixed point.
# - Starting from an arbitrary function $f$, the sequence $\{T^n[f]\}_{n=1}^{\infty}$ converges to such fixed point.
#
# For our purposes, this translates to:
# - Our equilibrium pricing function not only exists, but it is unique.
# - We can get arbitrarily close to the equilibrium pricing function by making some initial guess $f$ and applying the operator $T$ to it repeatedly. 
#
# The code below creates a representation of our model and implements a solution routine to find $P^*$. The main components of this routine are:
#
# - `priceOnePeriod`: this is operator $T$ from above. It takes a function $f$, computes $\beta\mathbb{E}_{t}\left[ \frac{u^{\prime}(d_{t+1})}{u^{\prime}(d_t)} (f(d_{t+1}) + d_{t+1}) \right]$ for a grid of $d_t$ values, and uses the result to construct a linear interpolator that approximates $T[f]$.
#
# - `solve`: this is our iterative solution procedure. It generates an initial guess $f$ and applies `priceOnePeriod` to it iteratively. At each application, it constructs a measure of how much the candidate pricing function changed. Once changes between successive iterations are small enough, it declares that the solution has converged.

# %% [markdown]
# # A computational representation of the problem and its solution.

# %% Preamble {"code_folding": [0]}
# Setup
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from HARK.utilities import CRRAutilityP
from HARK.distribution import Normal
from HARK.interpolation import LinearInterp, ConstantFunction

# %% Definitions {"code_folding": [0]}
# A class representing log-AR1 dividend processes.
class DivProcess:
    
    def __init__(self, alpha, shock_sd, shock_mean = 0.0, nApprox = 7):
        
        self.alpha = alpha
        self.shock_sd = shock_sd
        self.shock_mean = shock_mean
        self.nApprox = nApprox
        
        # Create a discrete approximation to the random shock
        self.ShkAppDstn = Normal(mu = shock_mean, sigma = shock_sd).approx(N = nApprox)
        
    def getLogdGrid(self, n = 100):
        '''
        A method for creating a reasonable grid for log-dividends.
        '''
        uncond_sd = self.shock_sd / np.sqrt(1 - self.alpha**2)
        uncond_mean = self.shock_mean/(1-self.alpha)
        logDGrid = np.linspace(-5*uncond_sd, 5*uncond_sd, n) + uncond_mean
        return(logDGrid)
        
# A class representing economies with Lucas' trees.
class LucasEconomy:
    '''
    A representation of an economy in which there are Lucas trees
    whose dividends' logarithm follows an AR1 process.
    '''
    def __init__(self, CRRA, DiscFac, DivProcess):
        
        self.CRRA = CRRA
        self.DiscFac = DiscFac
        self.DivProcess = DivProcess
        self.uP = lambda c: CRRAutilityP(c, self.CRRA)
        
    def priceOnePeriod(self, Pfunc_next, logDGrid):
        
        # Create 'tiled arrays' rows are state today, columns are value of
        # tomorrow's shock
        dGrid_N = len(logDGrid)
        shock_N = self.DivProcess.nApprox
        
        # Today's dividends
        logD_now = np.tile(np.reshape(logDGrid, (dGrid_N,1)),
                           (1,shock_N))
        d_now = np.exp(logD_now)
        
        # Tomorrow's dividends
        Shk_next = np.tile(self.DivProcess.ShkAppDstn.X,
                               (dGrid_N, 1))
        Shk_next_pmf = np.tile(self.DivProcess.ShkAppDstn.pmf,
                               (dGrid_N, 1))
        
        logD_next = self.DivProcess.alpha * logD_now + Shk_next
        d_next = np.exp(logD_next)
        
        # Tomorrow's prices
        P_next = Pfunc_next(logD_next)
        
        # Compute the RHS of the pricing equation, pre-expectation
        SDF = self.DiscFac * self.uP(d_next / d_now)
        Payoff_next = P_next + d_next
        
        # Take expectation and discount
        P_now = np.sum(SDF*Payoff_next*Shk_next_pmf, axis = 1, keepdims=True)
        
        # Create new interpolating price function
        Pfunc_now = LinearInterp(logDGrid, P_now.flatten(), lower_extrap=True)
        
        return(Pfunc_now)
        
    def solve(self, Pfunc_0 = None, logDGrid = None, tol = 1e-5, maxIter = 500, disp = False):
        
        # Initialize the norm
        norm = tol + 1
        
        # Initialize Pfunc if initial guess is not provided
        if Pfunc_0 is None:
            Pfunc_0 = ConstantFunction(0.0)
        
        # Create a grid for log-dividends if one is not provided
        if logDGrid is None:
            logDGrid = self.DivProcess.getLogdGrid()
        
        # Initialize function and compute prices on the grid
        Pf_0 = copy(Pfunc_0)
        P_0 = Pf_0(logDGrid)
        
        it = 0
        while norm > tol and it < maxIter:
            
            # Apply the pricing equation
            Pf_next = self.priceOnePeriod(Pf_0, logDGrid)
            # Find new prices on the grid
            P_next = Pf_next(logDGrid)
            # Measure the change between price vectors
            norm = np.linalg.norm(P_0 - P_next)
            # Update price function and vector
            Pf_0 = Pf_next
            P_0  = P_next
            it = it + 1
            # Print iteration information
            if disp:
                print('Iter:' + str(it) + '   Norm = '+ str(norm))
        
        if disp:
            if norm <= tol:
                print('Price function converged!')
            else:
                print('Maximum iterations exceeded!')
        
        self.EqlogPfun = Pf_0
        self.EqPfun = lambda d: self.EqlogPfun(np.log(d))


# %% [markdown]
# # Creating and solving an example economy
#
# An economy is fully specified by:
# - **The dividend process for the assets (trees)**: we assume that $\ln d_{t+1} = \alpha \ln d_t + \varepsilon_{t+1}$. We must create a dividend process specifying $\alpha$ and $\sigma_{\varepsilon}$.
# - **The coefficient of relative risk aversion (CRRA).**
# - **The time-discount factor ($\beta$).**

# %% Example {"code_folding": [0]}
# Create a log-AR1 process for dividends
DivProc = DivProcess(alpha = 0.90, shock_sd = 0.1)

# Create an example economy
economy = LucasEconomy(CRRA = 2, DiscFac = 0.95, DivProcess = DivProc)


# %% [markdown]
# Once created, the economy can be 'solved', which means finding the equilibrium price kernel. The distribution of dividends at period $t+1$ depends on the value of dividends at $t$, which also determines the resources agents have available to buy trees. Thus, $d_t$ is a state variable for the economy. The pricing function gives the price of trees that equates their demand and supply at every level of current dividends $d_t$.

# %% Solution {"code_folding": [0]}
# Solve the economy
economy.solve(disp = True)

# After the economy is solved, we can use its Equilibrium price function
d = 1
print('P({}) = {}'.format(d, economy.EqPfun(d)))


# %% [markdown]
# ## The effect of risk aversion.
#
# [The notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/AssetPricing/LucasAssetPrice/) discuss the surprising implication that an increase in the coefficient of relative risk aversion $\rho$ leads to higher prices for the risky trees! This is demonstrated below.

# %% {"code_folding": [0]}
# Create two economies with different risk aversion
Disc = 0.95
LowCrraEcon  = LucasEconomy(CRRA = 2, DiscFac = Disc, DivProcess = DivProc)
HighCrraEcon = LucasEconomy(CRRA = 4, DiscFac = Disc, DivProcess = DivProc)

# Solve both
LowCrraEcon.solve()
HighCrraEcon.solve()

# Plot the pricing functions for both
dGrid = np.linspace(0.5,2.5,30)
plt.plot(dGrid, LowCrraEcon.EqPfun(dGrid), label = 'Low CRRA')
plt.plot(dGrid, HighCrraEcon.EqPfun(dGrid), label = 'High CRRA')
plt.legend()
plt.xlabel('$d_t$')
plt.ylabel('$P_t$')

# %% [markdown]
# # Testing our analytical solutions

# %% [markdown]
# ## 1. Log-utility
#
# The lecture notes show that with log-utility (a CRRA of $1$), the pricing kernel has a closed form expression: $$P^*(d_t) = \frac{d_t}{\vartheta}$$.
#
# We now compare our numerical solution with this analytical expression.

# %% {"code_folding": [0]}
# Create an economy with log utility and the same dividend process from before
logUtilEcon = LucasEconomy(CRRA = 1, DiscFac = Disc, DivProcess = DivProc)
# Solve it
logUtilEcon.solve()

# Generate a function with our analytical solution
theta = 1/Disc - 1
aSol = lambda d: d/theta

# Get a grid for d over which to compare them
dGrid = np.exp(DivProc.getLogdGrid())

# Plot both
plt.plot(dGrid, aSol(dGrid), '*',label = 'Analytical solution')
plt.plot(dGrid, logUtilEcon.EqPfun(dGrid), label = 'Numerical solution')
plt.legend()
plt.xlabel('$d_t$')
plt.ylabel('$P^*(d_t)$')

# %% [markdown]
#  ## 2. I.I.D dividends
#  
#  We also found that, if $\ln d_{t+n}\sim \mathcal{N}(-\sigma^2/2, \sigma^2)$ for all $n$, the pricing kernel is exactly
#  \begin{equation*}
#  P^*(d_t) = d_t^\rho\times e^{\rho(\rho-1)\sigma^2/2}\frac{\beta}{1-\beta}.
#  \end{equation*}
#  
#  We now our numerical solution for this case.

# %% {"code_folding": [0]}
# Create an i.i.d. dividend process
shock_sd = 0.1
iidDivs = DivProcess(alpha = 0.0, shock_mean = -shock_sd**2/2, shock_sd = shock_sd)

# And an economy that embeds it
CRRA = 2
Disc = 0.9

iidEcon = LucasEconomy(CRRA = CRRA, DiscFac = Disc, DivProcess = iidDivs)
iidEcon.solve()

# Generate a function with our analytical solution
dTil = np.exp((shock_sd**2)/2*CRRA*(CRRA-1))
aSolIID = lambda d: d**CRRA * dTil * Disc/(1 - Disc)

# Get a grid for d over which to compare them
dGrid = np.exp(iidDivs.getLogdGrid())

# Plot both
plt.plot(dGrid, aSolIID(dGrid), '*',label = 'Analytical solution')
plt.plot(dGrid, iidEcon.EqPfun(dGrid), label = 'Numerical solution')
plt.legend()
plt.xlabel('$d_t$')
plt.ylabel('$P^*(d_t)$')

# %% [markdown]
# # Testing our approximation of the dividend process
#
# Hidden in the solution method implemented above is the fact that, in order to make expectations easy to compute, we discretize the random shock $\varepsilon_t$, which is to say, we create a discrete variable $\tilde{\varepsilon}$ that approximates the behavior of $\varepsilon_t$. This is done using a [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).
#
# A parameter for the numerical solution is the number of different values that we allow our discrete approximation $\tilde{\varepsilon}$ to take, $n^{\#}$. We would expect a higher $n^#$ to improve our solution, as the discrete approximation of $\varepsilon_t$ improves. We test this below.

# %% {"code_folding": [0]}
# Increase CRRA to make the effect of uncertainty more evident.
CRRA = 10
Disc = 0.9
shock_sd = 0.1
ns = [1,2,10]

# 
dTil = np.exp((shock_sd**2)/2*CRRA*(CRRA-1))
fact = dTil*Disc
aSolIID = lambda d: d**CRRA * dTil * Disc/(1 - Disc)

plt.figure()
for n in ns:
    iidDivs = DivProcess(alpha = 0.0, shock_mean = -shock_sd**2/2, shock_sd = shock_sd, nApprox = n)
    iidEcon = LucasEconomy(CRRA = CRRA, DiscFac = Disc, DivProcess = iidDivs)
    iidEcon.solve()
    plt.plot(dGrid, iidEcon.EqPfun(dGrid), label = 'Num.Sol. $n^\#$ = {}'.format(n))

# Plot both
plt.plot(dGrid, aSolIID(dGrid), '*',label = 'Analytical solution')
plt.legend()
plt.xlabel('$d_t$')
plt.ylabel('$P^*(d_t)$')

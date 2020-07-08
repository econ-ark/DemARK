# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,title
#     formats: ipynb,py:percent
#     rst2md: false
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
# # DCEGM Upper Envelope
# ## ["The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks"](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643)
#
# <p style="text-align: center;"><small><small><small>For the following badges: GitHub does not allow click-through redirects; right-click to get the link, then paste into navigation bar</small></small></small></p>
#
# [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks%2FDCEGM-Upper-Envelope.ipynb)
#
# [![Open in CoLab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb)
#
#
#
# This notebook provides a simple introduction to the upper envelope calculation in the "DCEGM" algorithm <cite data-cite="6202365/4F64GG8F"></cite>. It takes the EGM method proposed in <cite data-cite="6202365/HQ6H9JEI"></cite>, and extends it to the mixed choice (discrete and continuous) case. It handles various constraints. It works on a 1-dimensional problems.
#
# The main challenge in the types of models considered in DCEGM is, that the first order conditions to the Bellman equations are no longer sufficient to find an optimum.  Though, they are still necessary in a broad class of models. This means that our EGM step will give us (resource, consumption) pairs that do fulfill the FOCs, but that are sub-optimal (there's another consumption choices for the same initial resources that gives a higher value).
#
# Take a consumption model formulated as:
# $$
# \max_{\{c_t\}^T_{t=1}} \sum^T_{t=1}\beta^t\cdot u(c_t)
# $$
# given some initial condition on $x$ and some laws of motion for the states, though explicit references to states are omitted. Then, if we're in a class of models described in EGM
# , we can show that
# $$
# c_t = {u_{c}}^{-1}[E_t(u_c(c_{t+1}))]
# $$
# uniquely determines an optimal consumption today given the expected marginal utility of consuming  tomorrow. However, if there is a another choice in the choice set, and that choice is discrete, we get
# $$
# \max_{\{c_t, d_t\}^T_{t=1}} \sum^T_{t=1}\beta^t\cdot u(c_t, d_t)
# $$
# again given initial conditions and the laws of motion. Then, we can show that
# $$
# c_t = {u_{c}}^{-1}[E_t(u_c(c_{t+1}))]
# $$
# will produce solutions that are necessary but not sufficient. Note, that there is no explicit mentioning of the discrete choices in the expectation, but they obviously vary over the realized states in general. For the optimal consumption, it doesn't matter what the choice is exactly, only what expected marginal utility is tomorrow. The algorithm presented in [1] is designed to take advantage of models with this structure.
#
# To visualize the problem, consider the following pictures that show the output of an EGM step from the model in the REMARK [linkhere].

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt

# %%
# here for now, should be
# from HARK import discontools or whatever name is chosen
from HARK.interpolation import LinearInterp
from HARK.dcegm import calcSegments, calcMultilineEnvelope

# %%
m_common = np.linspace(0,1.0,100)
m_egm = np.array([0.0, 0.04, 0.25, 0.15, 0.1, 0.3, 0.6,0.5, 0.35, 0.6, 0.75,0.85])
c_egm = np.array([0.0, 0.03, 0.1, 0.07, 0.05, 0.36, 0.4, 0.6, 0.8, 0.9,0.9,0.9])
vt_egm = np.array( [0.0, 0.05, 0.1,0.04, 0.02,0.2, 0.7, 0.5, 0.2, 0.9, 1.0, 1.2])

# %%
plt.plot(m_egm, vt_egm)
plt.xlabel("resources")
plt.ylabel("transformed values")

# %%
plt.plot(m_egm, c_egm)
plt.xlabel("resources")
plt.ylabel("consumption")
plt.show()
# %% [markdown]
# The point of DCEGM is to realize, that the segments on the `(m, vt)` curve that are decreasing, cannot be optimal. This leaves us with a set of increasing line segments, as seen below (`dcegmSegments` is the function in HARK that calculates the breaks where the curve goes from increasing to decreasing).

# %%
rise, fall = calcSegments(m_egm, vt_egm)

# %% [markdown]
# In `rise` we have all the starting indices for the segments that are "good", that is `(m, vt)` draws an increasing curve.

# %%
rise

# %% [markdown]
# We see that `rise` has its first index at `0`, then again at `4`, and lastly at `8`. Let's look at `fall`.

# %%
fall

# %% [markdown]
# We see that the last segment is increasing (as the last element of `rise` is larger than the last element of `fall`), and we see that `len(fall)` is one larger than number of problematic segments in the plot. The index of the last point in `m_egm`/`c_egm`/`vt_egm` is added for convenience when we do the upper envelope step (and is also convenient below for drawing the segments!).
#
# We can use `fall` and `rise` to draw only the relevant segments that we will use to construct an upper envelope.

# %%
for j in range(len(fall)):
    idx = range(rise[j],fall[j]+1)
    plt.plot(m_egm[idx], vt_egm[idx])
plt.xlabel("resources")
plt.ylabel("transformed values")
plt.show()
# %% [markdown]
# Let us now use the `calcMultilineEnvelope` function to do the full DCEGM step: find segments and calculate upper envelope in one sweep.

# %%
m_upper, c_upper, v_upper = calcMultilineEnvelope(m_egm, c_egm, vt_egm, m_common)

# %%
for j in range(len(fall)):
    idx = range(rise[j],fall[j]+1)
    plt.plot(m_egm[idx], vt_egm[idx])
plt.plot(m_upper, v_upper, 'k')
plt.xlabel("resources")
plt.ylabel("transformed values")
plt.show()
# %% [markdown]
# And there we have it! These functions are the building blocks for univariate discrete choice modeling in HARK, so hopefully this little demo helped better understand what goes on under the hood, or it was a help if you're extending some existing class with a discrete choice.

# %% [markdown]
# # An example: writing a will?
#
# We now present a basic example to illustrate the use of the previous tools in solving dynamic optimization problems with discrete and continuous decisions.
#
# The model represents an agent that lives for three periods and decides how much of his resources to consume in each of them. On the second period, he must additionally decide whether to hire a lawyer to write a will. Having a will has the upside of allowing the agent to leave a bequest in his third and last period of life, which gives him utility, but has the downside that the lawyer will charge a fraction of his period 3 resources.
#
# I now present the model formally.
#
# # The third (last) period of life
#
# In the last period of life, the agent's problem is determined by his total amount of resources $m_3$ and a state variable $W$ that indicates whether he wrote a will ($W=1$) or not ($W=0$).
#
# ### The agent without a will
#
# An agent who does not have a will simply consumes all of his available resources. Therefore, his value and consumption functions will be:
#
# \begin{equation}
# V_3(m_3,W=0) = u(m_3)
# \end{equation}
#
# \begin{equation}
# c_3(m_3, W=0) = m_3
# \end{equation}
#
# Where $u(\cdot)$ gives the utility from consumption. We assume a CRRA specification $u(c) = \frac{c^{1-\rho}}{1-\rho}$.
#
# ### The agent with a will
#
# An agent who wrote a will decides how to allocate his available resources $m_3$ between his consumption and a bequest. We assume an additive specification for the utility of a given consumption-bequest combination that follows a particular case in [Carroll (2000)](http://www.econ2.jhu.edu/people/ccarroll/Why.pdf). The component of utility from leaving a bequest $x$ is assumed to be $\ln (x+1)$. Therefore, the agent's value function is
#
# \begin{equation}
# V_3(m_3, W=1) = \max_{0\leq c_3 \leq m_3} u(c_3) + \ln(m_3 - c_3 + 1)
# \end{equation}
#
# For ease of exposition we consider the case $\rho = 2$, where [Carroll (2000)](http://www.econ2.jhu.edu/people/ccarroll/Why.pdf) shows that the optimal consumption level is given by
#
# \begin{equation}
# c_3(m_3, W=1) = \min \left[m_3, \frac{-1 + \sqrt{1 + 4(m_3+1)}}{2} \right].
# \end{equation}
#
# The consumption shows that $m_3=1$ is the level of resources at which an important change of behavior occurs: agents leave bequests only for $m_3 > 1$. Since an important change of behavior happens at this point, we call it a 'kink-point' and add it to our grids.

# %%
# from HARK import discontools or whatever name is chosen
from HARK.interpolation import LinearInterp, calcLogSumChoiceProbs

# Params
y = 1
rra = 2 # This is fixed at 2! do not touch!
tau = 0.35
beta = 0.9
aGrid = np.linspace(0,8,100)

# Function defs
def crra(x, rra):
    if rra == 1:
        return np.log(x)
    return x**(1-rra)/(1-rra)

def crraP(x, rra):
    return x**(-rra)

def crraPInv(x, rra):
    return x**(-1/rra)

u     = lambda x: crra(x,rra)
uP    = lambda x: crraP(x, rra)
uPinv = lambda x: crraPInv(x, rra)

# Create a grid for market resources
mGrid = (aGrid-aGrid[0])*1.5
mGrid = mGrid[2:]

# %% [markdown]
# # The last period

# %%
# Agent without a will
m3_grid_no = mGrid
c3_grid_no = mGrid
v3_grid_no = u(c3_grid_no)

# Create functions
c3_no = LinearInterp(np.insert(m3_grid_no,0,0), np.insert(c3_grid_no,0,0))
v3_no = LinearInterp(m3_grid_no, v3_grid_no, lower_extrap = True)

# Agent with a will

# Define an auxiliary function with the analytical consumption expression
c3will = lambda m: np.minimum(m, -0.5 + 0.5*np.sqrt(1+4*(m+1)))

# Find the kink point
kink_m = 1.0
inds_below = mGrid < kink_m
inds_above = mGrid > kink_m

m3_grid_wi = np.concatenate([mGrid[inds_below],
                             np.array([kink_m]),
                             mGrid[inds_above]])

c3_grid_wi = c3will(m3_grid_wi)

c_above = c3will(mGrid[inds_above])
beq_above = mGrid[inds_above] - c3will(mGrid[inds_above])
v3_grid_wi = np.concatenate([u(mGrid[inds_below]),
                             u(np.array([kink_m])),
                             u(c_above) + np.log(1+beq_above)])

# Create functions
c3_wi = LinearInterp(np.insert(m3_grid_wi,0,0), np.insert(c3_grid_wi,0,0))
v3_wi = LinearInterp(m3_grid_wi, v3_grid_wi, lower_extrap = True)

plt.figure()

plt.plot(m3_grid_wi, v3_grid_wi, label = 'Will')
plt.plot(m3_grid_no, v3_grid_no, label = 'No Will')
plt.title('Period 3: Value functions')
plt.xlabel('Market resources')
plt.legend()
plt.show()

plt.plot(m3_grid_wi, c3_grid_wi, label = 'Will')
plt.plot(m3_grid_no, c3_grid_no, label = 'No Will')
plt.title('Period 3: Consumption Functions')
plt.xlabel('Market resources')
plt.legend()
plt.show()

# %% [markdown]
# # The second period
#
# On the second period, the agent takes his resources as given (the only state variable) and makes two decisions:
# - Whether to write a will or not.
# - What fraction of his resources to consume.
#
# These decisions can be seen as happening sequentially: the agent first decides whether to write a will or not, and then consumes optimally and taking his decision into account. Since we solve the model backwards in time, we first explore the consumption decision, conditional on the choice of writing a will or not.
#
# ## An agent who decides not to write a will
#
# After deciding not to write a will, an agent solves the optimization problem expressed in the following conditional value function
#
# \begin{equation}
# \begin{split}
# \nu (m_2|w=0) &= \max_{0\leq c \leq m_2} u(c) + \beta V_3(m_3,W=0)\\
# s.t.&\\
# m_3 &= m_2 - c + y
# \end{split} 
# \end{equation}
#
# We can approximate a solution to this problem through the method of endogenous gridpoints. This yields approximations to $\nu(\cdot|w=0)$ and $c_2(\cdot|w=0)$

# %%
# Second period, not writing a will

# Compute market resources at 3 with and without a will
m3_cond_nowi_g = aGrid + y
# Compute marginal value of assets in period 3 for each ammount of savings in 2
v3prime_no_g = uP(c3_no(m3_cond_nowi_g))
# Get consumption through EGM inversion of the euler equation
c2_cond_no_g = uPinv(beta*v3prime_no_g)

# Get beginning-of-period market resources
m2_cond_no_g = aGrid + c2_cond_no_g

# Compute value function
v2_cond_no_g = u(c2_cond_no_g) + beta*v3_no(m3_cond_nowi_g)

# Create interpolating value and consumption functions
v2_cond_no = LinearInterp(m2_cond_no_g, v2_cond_no_g, lower_extrap = True)
c2_cond_no = LinearInterp(np.insert(m2_cond_no_g,0,0), np.insert(c2_cond_no_g,0,0))

# %% [markdown]
# ## An agent who decides to write a will
#
# An agent who decides to write a will also solves for his consumption dinamically. We assume that the lawyer that helps the agent write his will takes some fraction $\tau$ of his total resources in period 3. Therefore, the evolution of resources is given by $m_3 = (1-\tau)(m_2 - c_2 + y)$. The conditional value function of the agent is therefore:
#
# \begin{equation}
# \begin{split}
# \nu (m_2|w=1) &= \max_{0\leq c \leq m_2} u(c) + \beta V_3(m_3,W=1)\\
# s.t.&\\
# m_3 &= (1-\tau)(m_2 - c + y)
# \end{split} 
# \end{equation}
#
# We also approximate a solution to this problem using the EGM. This yields approximations to $\nu(\cdot|w=1)$ and $c_2(\cdot|w=1)$.

# %%
# Second period, writing a will

# Compute market resources at 3 with and without a will
m3_cond_will_g = (1-tau)*(aGrid + y)
# Compute marginal value of assets in period 3 for each ammount of savings in 2
v3prime_wi_g = uP(c3_wi(m3_cond_will_g))
# Get consumption through EGM inversion of the euler equation
c2_cond_wi_g = uPinv(beta*(1-tau)*v3prime_wi_g)
# Get beginning-of-period market resources
m2_cond_wi_g = aGrid + c2_cond_wi_g

# Compute value function
v2_cond_wi_g = u(c2_cond_wi_g) + beta*v3_wi(m3_cond_will_g)

# Create interpolating value and consumption functions
v2_cond_wi = LinearInterp(m2_cond_wi_g, v2_cond_wi_g, lower_extrap = True)
c2_cond_wi = LinearInterp(np.insert(m2_cond_wi_g,0,0), np.insert(c2_cond_wi_g,0,0))

# %% [markdown]
# ## The decision whether to write a will or not
#
# With the conditional value functions at hand, we can now express and solve the decision of whether to write a will or not, and obtain the unconditional value and consumption functions.
#
# \begin{equation}
# V_2(m_2) = \max \{ \nu (m_2|w=0), \nu (m_2|w=1) \}
# \end{equation}
#
# \begin{equation}
# w^*(m_2) = \arg \max_{w \in \{0,1\}} \{ \nu (m_2|w=w) \}
# \end{equation}
#
# \begin{equation}
# c_2(m_2) = c_2(m_2|w=w^*(m_2))
# \end{equation}
#
# We now construct these objects.

# %%
# We use HARK's 'calcLogSumchoiceProbs' to compute the optimal
# will decision over our grid of market resources.
# The function also returns the unconditional value function
v2_grid, choices_2 = calcLogSumChoiceProbs(np.stack((v2_cond_wi(mGrid),
                                                     v2_cond_no(mGrid))),
                                           sigma = 0)

# Plot the optimal decision rule
plt.plot(mGrid, choices_2[0])
plt.title('$w^*(m)$')
plt.ylabel('Write will (1) or not (0)')
plt.xlabel('Market resources: m')
plt.show()

# With the decision rule we can get the unconditional consumption function
c2_grid = (choices_2*np.stack((c2_cond_wi(mGrid),c2_cond_no(mGrid)))).sum(axis=0)

v2 = LinearInterp(mGrid, v2_grid, lower_extrap = True)
c2 = LinearInterp(mGrid, c2_grid)

# Plot the conditional and unconditional value functions
plt.plot(mGrid, v2_cond_wi(mGrid), label = 'Cond. Will')
plt.plot(mGrid, v2_cond_no(mGrid), label = 'Cond. No will')
plt.plot(mGrid, v2(mGrid), 'k--',label = 'Uncond.')
plt.title('Period 2: Value Functions')
plt.xlabel('Market resources')
plt.legend()
plt.show()

# Plot the conditional and unconditiional consumption
# functions
plt.plot(mGrid, c2_cond_wi(mGrid), label = 'Cond. Will')
plt.plot(mGrid, c2_cond_no(mGrid), label = 'Cond. No will')
plt.plot(mGrid, c2(mGrid), 'k--',label = 'Uncond.')
plt.title('Period 2: Consumption Functions')
plt.xlabel('Market resources')
plt.legend()
plt.show()

# %% [markdown]
# # The first period
#
# In the first period, an agent simply observes his market resources and decides what fraction of them to consume. His problem is represented by the following value function
#
# \begin{equation}
# \begin{split}
# V (m_1) &= \max_{0\leq c \leq m_1} u(c) + \beta V_2(m_2)\\
# s.t.&\\
# m_2 &= m_1 - c + y.
# \end{split} 
# \end{equation}
#
# Although this looks like a simple problem, there are complications introduced by the kink in $V_2(\cdot)$, which is clearly visible in the plot from the previous block. Particularly, note that $V_2'(\cdot)$ and $c_2(\cdot)$ are not monotonic: there are now multiple points $m$ for which the slope of $V'_2(m)$ is equal. Thus, the Euler equation becomes a necessary but not sufficient condition for optimality and the traditional EGM inversion step can generate non-monotonic endogenous $m$ gridpoints.
#
# We now illustrate this phenomenon.

# %% Solve the first period
# EGM step

# Period 2 resources implied by the exogenous savings grid
m2_g = aGrid + y
# Envelope condition
v2prime_g = uP(c2(m2_g))
# Inversion of the euler equation
c1_g = uPinv(beta*v2prime_g)
# Endogenous gridpoints
m1_g = aGrid + c1_g
v1_g = u(c1_g) + beta*v2(m2_g)

plt.plot(m1_g)
plt.title('Endogenous gridpoints')
plt.xlabel('Position: i')
plt.ylabel('Endogenous grid point: $m_i$')
plt.show()


plt.plot(m1_g,v1_g)
plt.title('Value function at grid points')
plt.xlabel('Market resources: m')
plt.ylabel('Value function')
plt.show()

# %% [markdown]
# The previous cell applies the endogenous gridpoints method to the first period problem. The plots illustrate that the sequence of resulting endogenous gridpoints $\{m_i\}_{i=1}^N$ is not monotonic. This results in intervals of market resources over which we have multiple candidate values for the value function. This is the point where we must apply the upper envelope function illustrated above.

# %%
# Get the envelope
m_up_g, c_up_g, v_up_g = calcMultilineEnvelope(m1_g, c1_g, v1_g, mGrid)

# Show that there is a non-monothonicity and that the upper envelope fixes it
plt.plot(m1_g,v1_g, label = 'EGM Points')
plt.plot(m_up_g, v_up_g, 'k--', label = 'Upper Envelope')
plt.title('Period 1: Value function')
plt.legend()
plt.show()

plt.plot(m1_g,c1_g, label = 'EGM Points')
plt.plot(m_up_g,c_up_g,'k--', label = 'Upper Envelope')
plt.title('Period 1: Consumption function')
plt.legend()
plt.show()

# %% [markdown]
# # References
# [1] Iskhakov, F. , Jørgensen, T. H., Rust, J. and Schjerning, B. (2017), The endogenous grid method for discrete‐continuous dynamic choice models with (or without) taste shocks. Quantitative Economics, 8: 317-365. doi:10.3982/QE643
#
# [2] Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. Economics letters, 91(3), 312-320.
#
#

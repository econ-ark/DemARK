# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
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
#     version: 3.6.9
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: false
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
# ## Introduction: Keynes, Friedman, Modigliani

# %% {"code_folding": [0]}
# Some initial setup
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Dark2')

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import datetime as dt
import scipy.stats as stats
import statsmodels.formula.api as sm
from copy  import deepcopy

import pandas_datareader.data as web
# As of 09/03/2019 the latest available version of pandas-datareader
# has conflicts with the latest version of pandas. We temporarily fix
# this by loading data from files.
# This should not be necessary when pandas-datareader>0.7 becomes available.
from io import StringIO

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.utilities import plotFuncsDer, plotFuncs


# %% [markdown]
# ### 1. The Keynesian consumption function
#
# Keynes:
# 1. "The amount of aggregate consumption mainly depends on the amount of aggregate income."
# 1. It is a "fundamental psychological rule ... that when ... real income increases ... consumption [will increase], but by less than the increase in income."
# 1. More generally, "as a rule, a greater proportion of income ... is saved as real income increases."
#
# This can be formalized as:
#
# $
# \begin{eqnarray}
# c_t & = &  a_0 + a_{1}y_t
# \\ c_t - c_{t-1} & = & a_{1}(y_t - y_{t-1})
# \end{eqnarray}
# $
#
# for $a_0 > 0, a_1 < 1$
#

# %% [markdown]
# #### The Keynesian Consumption Function

# %% {"code_folding": [0]}
class KeynesianConsumer:
    """
    This class represents consumers that behave according to a
    Keynesian consumption function, representing them as a
    special case of HARK's PerfForesightConsumerType
    
    Methods:
    - cFunc: computes consumption/permanent income 
             given total income/permanent income.
    """
    
    def __init__(self):
        
        PFexample = PerfForesightConsumerType() # set up a consumer type and use default parameteres
        PFexample.cycles = 0 # Make this type have an infinite horizon
        PFexample.DiscFac = 0.05
        PFexample.PermGroFac = [0.7]

        PFexample.solve() # solve the consumer's problem
        PFexample.unpack('cFunc') # unpack the consumption function
        
        self.cFunc = PFexample.solution[0].cFunc
        self.a0 = self.cFunc(0)
        self.a1 = self.cFunc(1) - self.cFunc(0)


# %% {"code_folding": [0]}
# Plot cFunc(Y)=Y against the Keynesian consumption function
# Deaton-Friedman consumption function is a special case of perfect foresight model

# We first create a Keynesian consumer
KeynesianExample = KeynesianConsumer()

# and then plot its consumption function
income = np.linspace(0, 30, 20) # pick some income points
plt.figure(figsize=(9,6))
plt.plot(income, KeynesianExample.cFunc(income), label = 'Consumption function') #plot income versus the consumption
plt.plot(income, income, 'k--', label = 'C=Y')
plt.title('Consumption function')
plt.xlabel('Income (y)')
plt.ylabel('Normalized Consumption (c)')
plt.ylim(0, 20)
plt.legend()
plt.show()

# %% {"code_folding": [0]}
# This looks like the first of the three equations, consumption as a linear function of income!
# This means that even in a microfounded model (that HARK provides), the consumption function can match Keynes reduced form
# prediction (given the right parameterization).

# We can even find a_0 and a_1
a_0 = KeynesianExample.a0
a_1 = KeynesianExample.a1
print('a_0 is ' + str(a_0))
print('a_1 is ' +  str(a_1))

# %% [markdown]
# #### The Keynesian consumption function: Evidence

# %% [markdown]
# Aggregate Data:
#
# Long-term time-series estimates: $a_0$ close to zero, $a_1$ close to 1 (saving rate stable over time - Kuznets).<br>
# Short-term aggregate time-series estimates of change in consumption on change in income find $a_1 << 1$.<br>
# $c_t = a_0 + a_{1}y_t + a_{2}c_{t-1}$ finds significant $a_2$, near 1.

# %% {"code_folding": []}
# Lets have a look at some aggregate data

sdt = dt.datetime(1980, 1, 1) #set startdate
edt = dt.datetime (2017, 1, 1) #set end date
df = web.DataReader(["PCECC96", "DPIC96"], "fred", sdt, edt) #import the data from Fred
# Plot the data
plt.figure(figsize=(9,6))
plt.plot(df.DPIC96, df.PCECC96, 'go', markersize=3.0, label='Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(df.DPIC96, df.PCECC96)
plt.plot(df.DPIC96, intercept+slope*df.DPIC96, 'k-', label = 'Line of best fit')
plt.plot(df.DPIC96, df.DPIC96, 'k--', label = 'C=Y')
plt.xlabel('Income (y)')
plt.ylabel('Consumption (c)')
plt.legend()
plt.show()

print('a_0 is ' + str(intercept))
print('a_1 is ' +  str(slope))
# %%
# However, our consumption data is [non-stationary](https://www.reed.edu/economics/parker/312/tschapters/S13_Ch_4.pdf) and this drives the previous
# estimate.
df.DPIC96.plot()
plt.xlabel('Date')
plt.ylabel('Consumption (c)')

# %%
# Lets use our second equation to try to find an estimate of a_1

df_diff = df.diff() #create dataframe of differenced values

# Plot the data
plt.figure(figsize=(9,6))
plt.plot(df_diff.DPIC96, df_diff.PCECC96, 'go', markersize=3.0, label = 'Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(df_diff.DPIC96[1:], df_diff.PCECC96[1:]) # find line of best fit
plt.plot(df_diff.DPIC96[1:], intercept+slope*df_diff.DPIC96[1:], 'k-', label = 'Line of best fit')
plt.plot(np.array([-200, 200]), np.array([-200, 200]), 'k--', label = 'C=Y')
plt.xlabel('Change in income (dy)')
plt.ylabel('Change in consumption (dc)')
plt.legend()
plt.show()

print('a_1 is ' +  str(slope))

# %% [markdown]
# a_1 is now much lower, as we expected

# %% [markdown]
# ### Household Data:
#
# Cross-section plots of consumption and income: very large and significant $a_0$, $a_1$ maybe 0.5. <br>
#
# Further facts:
# 1. Black households save more than whites at a given income level.<br>
# 0. By income group:
#    * low-income: Implausibly large dissaving (spend 2 or 3 times income)
#    * high-income: Remarkably high saving

# %% [markdown]
# ### 2. Duesenberry

# %% [markdown]
# Habit formation may explain why $c_{t-1}$ affects $c_t$.<br>
# Relative Income Hypothesis suggests that you compare your consumption to consumption of ‘peers’.<br>
# May explain high saving rates of Black HHs.<br>
#
# Problems with Duesenberry: <br>
# No budget constraint<br>
# No serious treatment of intertemporal nature of saving

# %% [markdown]
# #### Dusenberry: Evidence

# %%
# Even if we control for income, past consumption seems to be significantly related to current consumption

df_habit = df.copy()
df_habit.columns = ['cons', 'inc']
df_habit['cons_m1'] = df.PCECC96.shift()
df_habit.dropna()

result = sm.ols(formula = "cons ~ inc + cons_m1", data=df_habit.dropna()).fit()
result.summary()


# %%
# The coefficient on lagged consumption is very significant.
# But regression may be statistically problematic for the usual [non-stationarity](https://towardsdatascience.com/stationarity-in-time-series-analysis-90c94f27322) reasons.

# %% [markdown]
# ### 3. Friedman's Permanent Income Hypothesis

# %% [markdown]
# $$c = p + u$$
# $$y = p + v$$
#
# We can try to test this theory across households. If we run a regression of the form:
# $$c_i = a_0 + a_{1}y_{i} + u_{i}$$
#
# And if Friedman is correct, and the "true" coefficient on permanent income $p$ is 1, then the coefficient on $y$ will be:
# $$a_1 = \frac{s^2_{p}}{(s^2_{v} + s^2_{p})}$$

# %% [markdown]
# #### Friedman's Permanent Income Hypothesis: HARK
#
# We begin by creating a class that class implements the Friedman PIH consumption function as a special case of the [Perfect Foresight CRRA](http://econ.jhu.edu/people/ccarroll/courses/choice/lecturenotes/consumption/PerfForesightCRRA) model.

# %% {"code_folding": [0]}
class FriedmanPIHConsumer:
    """
    This class represents consumers that behave according to
    Friedman's permanent income hypothesis, representing them as a
    special case of HARK's PerfForesightConsumerType
    
    Methods:
    - cFunc: computes consumption/permanent income 
             given total income/permanent income.
    """
    
    def __init__(self, Rfree=1.001, CRRA = 2):
        
        PFaux = PerfForesightConsumerType() # set up a consumer type and use default parameteres
        PFaux.cycles = 0 # Make this type have an infinite horizon
        PFaux.DiscFac = 1/Rfree
        PFaux.Rfree = Rfree
        PFaux.LivPrb = [1.0]
        PFaux.PermGroFac = [1.0]
        PFaux.CRRA = CRRA
        PFaux.solve() # solve the consumer's problem
        PFaux.unpack('cFunc') # unpack the consumption function
        
        self.cFunc = PFaux.solution[0].cFunc


# %%
# We can now create a PIH consumer
PIHexample = FriedmanPIHConsumer()

# Plot the perfect foresight consumption function
income = np.linspace(0, 10, 20) # pick some income points
plt.figure(figsize=(9,6))
plt.plot(income, PIHexample.cFunc(income), label = 'Consumption function') #plot income versus the consumption
plt.plot(income, income, 'k--', label = 'C=Y')
plt.title('Consumption function')
plt.xlabel('Normalized Income (y)')
plt.ylabel('Normalized Consumption (c)')
plt.legend()
plt.show()

# %% [markdown] {"code_folding": []}
# We can see that regardless of the income our agent receives, they consume their permanent income, which is normalized to 1.

# %% [markdown]
# We can also draw out some implications of the PIH that we can then test with evidence
#
# If we look at HH's who have very similar permanent incomes, we should get a small estimate of $a_1$, because $s^2_v$ is large relative to $s^2_p$.
#
# Lets simulate this using our HARK consumer.

# %%
# Permanent income has the same variance
# as transitory income.

perm_inc = np.random.normal(1., 0.1, 50)
trans_inc = np.random.normal(0.5, 0.1, 50)

total_inc = perm_inc + trans_inc

slope, intercept, r_value, p_value, std_err = stats.linregress(total_inc, PIHexample.cFunc(total_inc)*perm_inc)

plt.figure(figsize=(9,6))
plt.plot(total_inc, PIHexample.cFunc(total_inc)*perm_inc, 'go', label='Simulated data')
plt.plot(total_inc, intercept + slope*total_inc, 'k-', label='Line of best fit')
plt.plot(np.linspace(1, 2, 5), np.linspace(1, 2, 5), 'k--', label='C=Y')
plt.xlabel('Income (y)')
plt.ylabel('Consumption (c)')
plt.legend()
plt.ylim(0, 2)
plt.xlim(1.1, 1.9)
plt.show()

print('a_0 is ' + str(intercept))
print('a_1 is ' +  str(slope))

# %%
# Permanent income with higher variance

perm_inc = np.random.normal(1., 0.5, 50)
trans_inc = np.random.normal(0.5, 0.1, 50)

total_inc = perm_inc + trans_inc

slope, intercept, r_value, p_value, std_err = stats.linregress(total_inc, PIHexample.cFunc(total_inc)*perm_inc)

plt.figure(figsize=(9,6))
plt.plot(total_inc, PIHexample.cFunc(total_inc)*perm_inc, 'go', label='Simulated data')
plt.plot(total_inc, intercept + slope*total_inc, 'k-', label='Line of best fit')
plt.plot(np.linspace(0, 2, 5), np.linspace(0, 2, 5), 'k--', label='C=Y')
plt.xlabel('Income (y)')
plt.ylabel('Consumption (c)')
plt.legend()
plt.ylim(0, 2)
plt.show()

print('a_0 is ' + str(intercept))
print('a_1 is ' +  str(slope))

# %% [markdown]
# We can see that as we increase the variance of permanent income, the estimate of a_1 rises

# %% [markdown]
# #### Friedman's Permanent Income Hypothesis: Evidence

# %% [markdown]
# We can now consider the empirical evidence for the claims our HARK model made about the PIH.
#
# If we take a long time series, then the differences in permanent income should be the main driver of the variance in total income. This implies that a_1 should be high.
#
# If we take higher frequency time series (or cross sectional data), transitory shocks should dominate, and our estimate of a_1 should be lower.

# %% {"code_folding": [0]}
# Lets use the data from FRED that we used before.

# Using quarterly data (copying from above), we had:

plt.figure(figsize=(9,6))
plt.plot(df_diff.DPIC96, df_diff.PCECC96, 'go', markersize=3.0, label = 'Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(df_diff.DPIC96[1:], df_diff.PCECC96[1:]) # find line of best fit
plt.plot(df_diff.DPIC96[1:], intercept+slope*df_diff.DPIC96[1:], 'k-', label = 'Line of best fit')
plt.plot(np.array([-200, 200]), np.array([-200, 200]), 'k--', label = 'C=Y')
plt.xlabel('Change in income (dy)')
plt.ylabel('Change in consumption (dc)')
plt.legend()
plt.show()

print('a_1 is ' +  str(slope))

# %% {"code_folding": [0]}
# Using annual data

sdt = dt.datetime(1980, 1, 1) #set startdate
edt = dt.datetime (2017, 1, 1) #set end date
df_an = web.DataReader(["PCECCA", "A067RX1A020NBEA"], "fred", sdt, edt) #import the annual data from Fred

df_an_diff = df_an.diff()
df_an_diff.columns = ['cons', 'inc']

plt.figure(figsize=(9,6))
plt.plot(df_an_diff.inc, df_an_diff.cons, 'go', label='Data')
slope, intercept, r_value, p_value, std_err = stats.linregress(df_an_diff.inc[1:], df_an_diff.cons[1:]) # find line of best fit
plt.plot(df_an_diff.inc[1:], intercept+slope*df_an_diff.inc[1:], 'k-', label='Line of best fit')
plt.plot(np.linspace(-100, 500, 3), np.linspace(-100, 500, 3), 'k--', label='C=Y')
plt.legend()
plt.xlabel('Change in income (dy)')
plt.ylabel('Change in consumption (dc)')
plt.show()

print('a_0 is ' + str(intercept))
print('a_1 is ' +  str(slope))

# %% [markdown]
# The estimate of $a_1$ using the annual data is much higher because permanent income is playing a much more important role in explaining the variation in consumption.

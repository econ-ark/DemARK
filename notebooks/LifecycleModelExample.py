# ---
# jupyter:
#   cite2c:
#     citations:
#       6202365/7MR8GUVS:
#         DOI: 10.3982/QE694
#         URL: https://onlinelibrary.wiley.com/doi/abs/10.3982/QE694
#         abstract: In a model calibrated to match micro- and macroeconomic evidence
#           on household income dynamics, we show that a modest degree of heterogeneity
#           in household preferences or beliefs is sufficient to match empirical measures
#           of wealth inequality in the United States. The heterogeneity-augmented model's
#           predictions are consistent with microeconomic evidence that suggests that
#           the annual marginal propensity to consume (MPC) is much larger than the
#           roughly 0.04 implied by commonly used macroeconomic models (even ones including
#           some heterogeneity). The high MPC arises because many consumers hold little
#           wealth despite having a strong precautionary motive. Our model also plausibly
#           predicts that the aggregate MPC can differ greatly depending on how the
#           shock is distributed across households (depending, e.g., on their wealth,
#           or employment status).
#         accessed:
#           day: 5
#           month: 2
#           year: 2019
#         author:
#         - family: Carroll
#           given: Christopher
#         - family: Slacalek
#           given: Jiri
#         - family: Tokuoka
#           given: Kiichi
#         - family: White
#           given: Matthew N.
#         container-title: Quantitative Economics
#         id: 6202365/7MR8GUVS
#         issue: '3'
#         issued:
#           year: 2017
#         language: en
#         note: 'Citation Key: carrollDistributionWealthMarginal2017'
#         page: 977-1020
#         page-first: '977'
#         title: The distribution of wealth and the marginal propensity to consume
#         type: article-journal
#         volume: '8'
#       6202365/B9BGV9W3:
#         URL: http://www.nber.org/papers/w22822
#         abstract: "We provide a systematic analysis of the properties of individual\
#           \ returns to wealth using twenty years of population data from Norway\u2019\
#           s administrative tax records. We document a number of novel results. First,\
#           \ in a given cross-section, individuals earn markedly different returns\
#           \ on their assets, with a difference of 500 basis points between the 10th\
#           \ and the 90th percentile. Second, heterogeneity in returns does not arise\
#           \ merely from differences in the allocation of wealth between safe and risky\
#           \ assets: returns are heterogeneous even within asset classes. Third, returns\
#           \ are positively correlated with wealth. Fourth, returns have an individual\
#           \ permanent component that accounts for 60% of the explained variation.\
#           \ Fifth, for wealth below the 95th percentile, the individual permanent\
#           \ component accounts for the bulk of the correlation between returns and\
#           \ wealth; the correlation at the top reflects both compensation for risk\
#           \ and the correlation of wealth with the individual permanent component.\
#           \ Finally, the permanent component of the return to wealth is also (mildly)\
#           \ correlated across generations. We discuss the implications of these findings\
#           \ for several strands of the wealth inequality debate."
#         accessed:
#           day: 17
#           month: 3
#           year: 2019
#         author:
#         - family: Fagereng
#           given: Andreas
#         - family: Guiso
#           given: Luigi
#         - family: Malacrino
#           given: Davide
#         - family: Pistaferri
#           given: Luigi
#         genre: Working Paper
#         id: 6202365/B9BGV9W3
#         issued:
#           month: 11
#           year: 2016
#         note: 'DOI: 10.3386/w22822'
#         number: '22822'
#         publisher: National Bureau of Economic Research
#         title: Heterogeneity and Persistence in Returns to Wealth
#         type: report
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
# # The Distribution of Assets By Age
#
# National registry data on income and wealth from Scandinavian countries has recently become available (with a lot of security) to some (lucky!) researchers.   These data offer a uniquely powerful tool for testing (and improving) our models of consumption and saving behavior over the life cycle.
#
#
# But as of this writing (in March of 2019), the data are so new that there do not seem to be any published attempts to compare them to the implications of formal models.
#
# This notebook is an example of how one could counstruct a life cycle model with the HARK toolkit that would make predictions about the model analogues of the raw data statistics that are available.  
#
# For example, the papers have shown information about the growth rate of assets at different ages over the life cycle.  Here, we show how (under a given parameterization) we could produce the life cycle model's prediction about the distribution of assets at age 65 and age 66, and the growth rate between 65 and 66. 
#
# The parameters of the model have not been optmized to match features of the Norwegian data; a first step in "structural" estimation would be to calibrate the inputs to the model (like the profile of income over the life cycle, and the magnitude of income shocks), and then to find the values of parameters like the time preference rate that allow the model to fit the data best.
#
# An interesting question is whether this exercise will suggest that it is necessary to allow for _ex ante_ heterogeneity in such preference parameters.
#
# This seems likely; a paper by [<cite data-cite="6202365/7MR8GUVS"></cite>](http://econ.jhu.edu/people/ccarroll/papers/cstwMPC) (all of whose results were constructed using the HARK toolkit) finds that, if all other parameters (e.g., rates of return on savings) are the same, models of this kind require substantial heterogeneity in preferences to generate the degree of inequality in U.S. data.
#
# But in one of the many new and interesting findings from the Norwegian data, <cite data-cite="6202365/B9BGV9W3"></cite> have shown that there is substantial heterogeneity in rates of return, even on wealth held in public markets.  
#
# [Derin Aksit](https://github.com/econ-ark/REMARK) has shown that the degree of time preference heterogeneity needed to match observed inequality is considerably less when rate-of-return heterogeneity is calibrated to match these data.

# %%
# Initial imports and notebook setup, click arrow to show

import HARK.ConsumptionSaving.ConsIndShockModel as Model        # The consumption-saving micro model
import HARK.SolvingMicroDSOPs.EstimationParameters as Params    # Parameters for the consumer type and the estimation
from HARK.utilities import plotFuncsDer, plotFuncs              # Some tools

# %%
# Set up default values for CRRA, DiscFac, and simulation variables in the dictionary 
Params.init_consumer_objects["CRRA"]= 2.00            # Default coefficient of relative risk aversion (rho)
Params.init_consumer_objects["DiscFac"]= 0.97         # Default intertemporal discount factor (beta)
Params.init_consumer_objects["PermGroFacAgg"]= 1.0    # Aggregate permanent income growth factor 
Params.init_consumer_objects["aNrmInitMean"]= -10.0   # Mean of log initial assets 
Params.init_consumer_objects["aNrmInitStd"]= 1.0      # Standard deviation of log initial assets
Params.init_consumer_objects["pLvlInitMean"]= 0.0     # Mean of log initial permanent income 
Params.init_consumer_objects["pLvlInitStd"]= 0.0      # Standard deviation of log initial permanent income

# %%
# Make a lifecycle consumer to be used for estimation
LifeCyclePop = Model.IndShockConsumerType(**Params.init_consumer_objects)

# %%
# Solve and simulate the model (ignore the "warning" message)
LifeCyclePop.solve()                            # Obtain consumption rules by age 
LifeCyclePop.unpackcFunc()                      # Expose the consumption rules
LifeCyclePop.track_vars = ['aNrmNow','pLvlNow'] # Which variables are we interested in
LifeCyclePop.T_sim = 120                        # Nobody lives to be older than 145 years (=25+120)
LifeCyclePop.initializeSim()                    # Construct the age-25 distribution of income and assets
LifeCyclePop.simulate()                         # Simulate a population behaving according to this model

# %%
# Plot the consumption functions during working life

print('Consumption as a function of market resources while working:')
mMin = min([LifeCyclePop.solution[t].mNrmMin for t in range(LifeCyclePop.T_cycle)])
plotFuncs(LifeCyclePop.cFunc[:LifeCyclePop.T_retire],mMin,5)

# %%
# Construct the level of assets A from a*p where a is the ratio to permanent income p
LifeCyclePop.aLvlNow_hist = LifeCyclePop.aNrmNow_hist*LifeCyclePop.pLvlNow_hist
aGro41=LifeCyclePop.aLvlNow_hist[41]/LifeCyclePop.aLvlNow_hist[40]
aGro41NoU=aGro41[aGro41[:]>0.2] # Throw out extreme outliers

# %%
# Plot the distribution of growth rates of wealth between age 65 and 66 (=25 + 41)
from matplotlib import pyplot as plt
n, bins, patches = plt.hist(aGro41NoU,50,density=True)

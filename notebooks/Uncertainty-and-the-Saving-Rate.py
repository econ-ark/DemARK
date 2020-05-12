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
# # Uncertainty and Saving in Partial Equilibrium
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks%2FUncertainty-and-the-Saving-Rate.ipynb)
#
# Saving rates vary widely across countries, but there is no consensus about the main causes of those differences.
#
# One commonly mentioned factor is differences across countries in the degree of uncertainty that individuals face, which should induce different amounts of precautionary saving.
#
# Uncertainty might differ for "fundamental" reasons, having to do with, say, the volatility of demand for the goods and services supplied by the country, or might differ as a result of economic policies, such as the strucutre of the social insurance system.
#
# A challenge in evaluating the importance of precautionary motives for cross-country saving differences has been a lack of consensus about what measures of uncertainty ought, in principle, to be the right ones to look at in any attempt to measure a relationship between uncertainty and saving.
#
# This notebook uses [a standard model](https://econ.jhu.edu/people/ccarroll/papers/cstwMPC) <cite data-cite="6202365/7MR8GUVS"></cite> to construct a theoretical benchmark for the relationship of saving to two kinds of uncertainty: Permanent shocks and transitory shocks to income.  
#
# Conclusions:
# 1. The model implies a close to linear relationship between the variance of either kind of shock (transitory or permanent) and the saving rate
# 2. The _slope_ of that relationship is much steeper for permanent than for transitory shocks
#    * Over ranges of values calibrated to be representative of microeconomically plausible magnitudes
#
# Thus, the quantitative theory of precautionary saving says that the principal determinant of precautionary saving should be the magnitude of permanent (or highly persistent) shocks to income.
#
# (Because the result was obtained in a partial equilibrium model, the conclusion applies also to attempts to measure the magnitude of precautionary saving across groups of people who face different degrees of uncertainty within a country).
#
# @authors: Derin Aksit, Tongli Zhang, Christopher Carroll

# %% {"code_folding": [0, 11]}
# Boring non-HARK setup stuff

Generator = True # This notebook can be used as a source for generating derivative notebooks
nb_name = 'Uncertainty-and-the-Saving-Rate'

# This is a jupytext paired notebook that autogenerates BufferStockTheory.py
# which can be executed from a terminal command line via "ipython BufferStockTheory.py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

from IPython import get_ipython # In case it was run from python instead of ipython
def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    get_ipython().run_line_magic('matplotlib', 'auto')
    print('You appear to be running from a terminal')
    print('By default, figures will appear one by one')
    print('Close the visible figure in order to see the next one')

# Import the plot-figure library matplotlib

import matplotlib.pyplot as plt

# In order to use LaTeX to manage all text layout in our figures, we import rc settings from matplotlib.
from matplotlib import rc
plt.rc('font', family='serif')

# LaTeX is huge and takes forever to install on mybinder
# so if it is not installed then do not use it 
from distutils.spawn import find_executable
iflatexExists=False
if find_executable('latex'):
    iflatexExists=True
    
plt.rc('font', family='serif')
plt.rc('text', usetex=iflatexExists)

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings
warnings.filterwarnings("ignore")

import os
from copy import copy, deepcopy

# Define (and create, if necessary) the figures directory "Figures"
if Generator:
    nb_file_path = os.path.dirname(os.path.abspath(nb_name+".ipynb")) # Find pathname to this file:
    FigDir = os.path.join(nb_file_path,"Figures/") # LaTeX document assumes figures will be here
#    FigDir = os.path.join(nb_file_path,"/tmp/Figures/") # Uncomment to make figures outside of git path
    if not os.path.exists(FigDir):
        os.makedirs(FigDir)

from copy import deepcopy
from scipy.optimize import golden, brentq
from time import time
import numpy as np
import scipy as sp

# %% {"code_folding": [0]}
# Import HARK tools and cstwMPC parameter values
from HARK.utilities import plotFuncsDer, plotFuncs
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.ConsumptionSaving.ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy
from HARK.datasets import load_SCF_wealth_weights
from HARK.utilities import getLorenzShares

#import HARK.cstwMPC.cstwMPC as cstwMPC

SCF_wealth, SCF_weights = load_SCF_wealth_weights()
# Which points of the Lorenz curve to match in beta-dist (must be in (0,1))
percentiles_to_match = [0.2, 0.4, 0.6, 0.8]    

# Double the default value of variance
# Params.init_infinite['PermShkStd'] = [i*2 for i in Params.init_infinite['PermShkStd']]

# %% {"code_folding": [0]}
# Setup stuff for general equilibrium version

# Set targets for K/Y and the Lorenz curve
lorenz_target = getLorenzShares(SCF_wealth,
                                weights= SCF_weights,
                                percentiles= percentiles_to_match)

lorenz_long_data = np.hstack((np.array(0.0),\
                              getLorenzShares(SCF_wealth,
                                              weights=SCF_weights,
                                              percentiles=np.arange(0.01,1.0,0.01).tolist()),np.array(1.0)))
KY_target = 10.26

# %% {"code_folding": [0]}
# Setup and calibration of the agent types

# Define a dictionary with calibrated parameters
cstwMPC_init_infinite = {
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
    "aXtraMax":40,  # Maximum end-of-period assets in grid
    "aXtraCount":32,  # Number of points in assets grid
    "aXtraExtra":[None],
    "aXtraNestFac":3,  # Number of times to 'exponentially nest' when constructing assets grid
    "LivPrb":[1.0 - 1.0/160.0],  # Survival probability
    "DiscFac":0.97,             # Default intertemporal discount factor; dummy value, will be overwritten
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
    'AgentCount':10000,
}

# The parameter values below are taken from
# http://econ.jhu.edu/people/ccarroll/papers/cjSOE/#calibration

init_cjSOE                  = cstwMPC_init_infinite # Get default values of all parameters
# Now change some of the parameters for the individual's problem to those of cjSOE
init_cjSOE['CRRA']          = 2
init_cjSOE['Rfree']         = 1.04**0.25
init_cjSOE['PermGroFac']    = [1.01**0.25] # Indiviual-specific income growth (from experience, e.g.)
init_cjSOE['PermGroFacAgg'] = 1.04**0.25   # Aggregate productivity growth 
init_cjSOE['LivPrb']        = [0.95**0.25] # Matches a short working life 

PopGroFac_cjSOE = [1.01**0.25] # Irrelevant to the individual's choice; attach later to "market" economy object

# Instantiate the baseline agent type with the parameters defined above
BaselineType = AggShockConsumerType(**init_cjSOE)
BaselineType.AgeDstn = np.array(1.0) # Fix the age distribution of agents

# Make desired number of agent types (to capture ex-ante heterogeneity)
EstimationAgentList = []
for n in range(1):
    EstimationAgentList.append(deepcopy(BaselineType))
    EstimationAgentList[n].seed = n  # Give every instance a different seed

# %% {"code_folding": [0]}
# Make an economy for the consumers to live in

init_market = {'LorenzBool': False,
               'ManyStatsBool': False,
               'ignore_periods':0,    # Will get overwritten
               'PopGroFac':0.0,       # Will get overwritten
               'T_retire':0,          # Will get overwritten
               'TypeWeights':[],      # Will get overwritten
               'Population': 10000,
               'act_T':0,             # Will get overwritten
               'IncUnemp':0.15,
               'cutoffs':[(0.99,1),(0.9,1),(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0.0,0.2)],
               'LorenzPercentiles':percentiles_to_match,
               'AggShockBool':False
               }

EstimationEconomy                = CobbDouglasEconomy(init_market)
EstimationEconomy.print_parallel_error_once = True  # Avoids a bug in the code

EstimationEconomy.agents         = EstimationAgentList
EstimationEconomy.act_T          = 1200  # How many periods of history are good enough for "steady state"

# %% {"code_folding": [0]}
# Uninteresting parameters that also need to be set 
EstimationEconomy.KYratioTarget  = KY_target
EstimationEconomy.LorenzTarget   = lorenz_target
EstimationEconomy.LorenzData     = lorenz_long_data
EstimationEconomy.PopGroFac      = PopGroFac_cjSOE # Population growth characterizes the entire economy
EstimationEconomy.ignore_periods = 400 # Presample periods

#Display statistics about the estimated model (or not)
EstimationEconomy.LorenzBool     = False
EstimationEconomy.ManyStatsBool  = False
EstimationEconomy.TypeWeight     = [1.0]


# %%
def distributeParams(self,param_name,param_count,center,spread,dist_type):
        '''
        Distributes heterogeneous values of one parameter to the AgentTypes in self.agents.
        Parameters
        ----------
        param_name : string
            Name of the parameter to be assigned.
        param_count : int
            Number of different values the parameter will take on.
        center : float
            A measure of centrality for the distribution of the parameter.
        spread : float
            A measure of spread or diffusion for the distribution of the parameter.
        dist_type : string
            The type of distribution to be used.  Can be "lognormal" or "uniform" (can expand).
        Returns
        -------
        None
        '''
        # Get a list of discrete values for the parameter
        if dist_type == 'uniform':
            # If uniform, center is middle of distribution, spread is distance to either edge
            param_dist = approxUniform(N=param_count,bot=center-spread,top=center+spread)
        elif dist_type == 'lognormal':
            # If lognormal, center is the mean and spread is the standard deviation (in log)
            tail_N = 3
            param_dist = approxLognormal(N=param_count-tail_N,mu=np.log(center)-0.5*spread**2,sigma=spread,tail_N=tail_N,tail_bound=[0.0,0.9], tail_order=np.e)

        # Distribute the parameters to the various types, assigning consecutive types the same
        # value if there are more types than values
        replication_factor = len(self.agents) // param_count 
            # Note: the double division is intenger division in Python 3 and 2.7, this makes it explicit
        j = 0
        b = 0
        while j < len(self.agents):
            for n in range(replication_factor):
                self.agents[j](AgentCount = int(self.Population*param_dist[0][b]*self.TypeWeight[n]))
                exec('self.agents[j](' + param_name + '= param_dist[1][b])')
                j += 1
            b += 1
            
EstimationEconomy.distributeParams = distributeParams

# %% {"code_folding": [0]}
# Construct the economy at date 0
EstimationEconomy.distributeParams( # Construct consumer types whose heterogeneity is in the given parameter
    'DiscFac',
     7,# How many different types of consumer are there 
    center_estimate,       # Increase patience slightly vs cstwMPC so that maximum saving rate is higher
    spread_estimate,       # How much difference is there across consumers
    Params.dist_type)      # Default is for a uniform distribution


# %% {"code_folding": [0]}
# Function to calculate the saving rate of a cstw economy
def calcSavRte(Economy,ParamToChange,NewVals):
    '''
    Calculates the saving rate as income minus consumption divided by income.
    
    Parameters
    ----------
    Economy : [cstwMPCmarket] 
        A fully-parameterized instance of a cstwMPCmarket economy
    ParamToChange : string
        Name of the parameter that should be varied from the original value in Economy
    NewVals : [float] or [list]
        The alternative value (or list of values) that the parameter should take

    Returns
    -------
    savRte : [float]
        The aggregate saving rate in the last year of the generated history
    '''
    for NewVal in NewVals:
        if ParamToChange in ["PermShkStd","TranShkStd"]:
            ThisVal = [NewVal]
        else:
            ThisVal = NewVal # If they asked to change something else, assume it's a scalar
            
        for j in range(len(Economy.agents)): # For each agent, set the new parameter value
            setattr(Economy.agents[j],ParamToChange,ThisVal)
            cstwMPC.cstwMPCagent.updateIncomeProcess(Economy.agents[j]) 
        
        Economy.solve()
        
        C_NrmNow=[]
        A_NrmNow=[]
        M_NrmNow=[]
        for j in range (len(Economy.agents)): # Combine the results across all the agents
            C_NrmNow=np.hstack((C_NrmNow,Economy.agents[j].cNrmNow))
            A_NrmNow=np.hstack((A_NrmNow,Economy.agents[j].aNrmNow))
            M_NrmNow=np.hstack((M_NrmNow,Economy.agents[j].mNrmNow))
        CAgg=np.sum(np.hstack(Economy.pLvlNow)*C_NrmNow) # cNrm times pLvl = level of c; sum these for CAgg
        AAgg=np.sum(np.hstack(Economy.pLvlNow)*A_NrmNow) # Aggregate Assets
        MAgg=np.sum(np.hstack(Economy.pLvlNow)*M_NrmNow) # Aggregate Market Resources
        YAgg=np.sum(np.hstack(Economy.pLvlNow)*np.hstack(Economy.TranShkNow)) # Aggregate Labor Income
        BAgg=MAgg-YAgg # Aggregate "Bank Balances" (at beginning of period; before consumption decision)
        IncAgg=(BaselineType.Rfree-1)*BAgg+YAgg # Interest income plus noninterest income
        savRte=(IncAgg-CAgg)/IncAgg # Unspent income divided by the level of income
        return savRte


# %% {"code_folding": [0]}
# Function to plot relationship between x and y; x is the parameter varied and y is saving rate
def plotReg(x,y,xMin,xMax,yMin,yMax,xLbl,yLbl,Title,fileName):
    # Result_data_path = os.path.join(Folder_path,'SavingVSPermShr_Youth_MPC_15.png')
    plt.ylabel(yLbl)
    plt.xlabel(xLbl)
    plt.title(Title)
    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax)
    plt.scatter(x,y)
    # Draw the linear fitted line
    m, b = np.polyfit(x, y, 1)
#    plt.plot(x, m*np.asarray(x) + b, '-')
    if Generator:
        plt.savefig(FigDir + nb_name + '-' + fileName + '.png')
        plt.savefig(FigDir + nb_name + '-' + fileName + '.svg')
        plt.savefig(FigDir + nb_name + '-' + fileName + '.pdf')
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
    print('Slope=' + str(slope) + ', intercept=' + str(intercept) + ', r_value=' + str(r_value) + ', p_value=' + str(p_value)+', std=' + str(std_err))


# %% {"code_folding": [0]}
# Proportion of base value for uncertainty parameter to take (up to 1 = 100 percent)
# Do not go above one to avoid having to worry about whether the most patient consumer violates the 
# Growth Impatience Condition (https://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#GIC)
bottom=0.5
points=np.arange(bottom,1.+0.025,0.025)

# %% {"code_folding": [0]}
# Calculate variance of permanent shock vs saving measures
savRteList = []
KtoYList   = []
pVarList   = []
pVarBase = BaselineType.PermShkStd[0] ** 2
for pVar in points * pVarBase:
    pVarList.append(pVar) # Variance is square of standard deviation
    pStd = pVar ** 0.5
#    print(pStd)
    savRteList.append(calcSavRte(EstimationEconomy,"PermShkStd",[pStd]))
    KtoYList.append(0.25*np.mean(np.array(EstimationEconomy.KtoYnow_hist)[EstimationEconomy.ignore_periods:]))

# %% {"code_folding": [0]}
# Calculate how much net worth shrinks when permanent variance is halved 
ShrinksBy = KtoYList[1]/KtoYList[-1]
print('Halving the magnitude of the permanent variance causes target wealth to fall to %1.3f' % ShrinksBy)
print('of its original value.')

# %% {"code_folding": [0]}
# Plot pVar vs saving measures
plotReg(pVarList,savRteList,
        xMin=pVarList[1]-0.0002,xMax=pVarList[-1]+0.0002,yMin=savRteList[1]-0.01,yMax=savRteList[-1]+0.01,
        xLbl=r'Variance of Permanent Shocks, $\sigma^{2}_{\psi}$',
        yLbl='Aggregate Saving Rate',
        Title='Uncertainty vs Saving',
        fileName='savRtevsPermShkVar'
       )
plt.show(block=False)
plotReg(pVarList,KtoYList,
        xMin=pVarList[1]-0.0002,xMax=pVarList[-1]+0.0002,yMin=1.7,yMax=KtoYList[-1]+0.1,
        xLbl=r'Variance of Permanent Shocks, $\sigma^{2}_{\psi}$',
        yLbl='Net Worth/Income',
        Title='Uncertainty vs Net Worth Ratio',
        fileName='BvsPermShkVar'
       )
plt.ylabel('Net Worth/Income')
plt.xlabel(r'Variance of Permanent Shocks, $\sigma^{2}_{\psi}$')
plt.title('Uncertainty vs Net Worth Ratio',fontsize=16)
plt.xlim(pVarList[1]-0.0002,pVarList[-1]+0.0002)
plt.ylim(1.6,KtoYList[-1]+0.1)
plt.scatter(pVarList,KtoYList)
plt.xticks([pVarList[1],pVarList[-1]],[r'$\bar{\sigma}^{2}_{\psi}/2$',r'$\bar{\sigma}^{2}_{\psi}$'])
fileName='BvsPermShkVar'
if Generator:
        plt.savefig(FigDir + nb_name + '-' + fileName + '.png')
        plt.savefig(FigDir + nb_name + '-' + fileName + '.svg')
        plt.savefig(FigDir + nb_name + '-' + fileName + '.pdf')
plt.show(block=False)                

# %% {"code_folding": [0]}
# Calculate variance of transitory shock vs saving measures
# Restore benchmark solution
EstimationEconomy.distributeParams( # Construct consumer types whose heterogeneity is in the given parameter
    'DiscFac',
    Params.pref_type_count,# How many different types of consumer are there 
    center_estimate,       # Increase patience slightly vs cstwMPC so that maximum saving rate is higher
    spread_estimate,       # How much difference is there across consumers
    Params.dist_type)      # Default is for a uniform distribution
EstimationEconomy.solve()

savRteList_Tran = []
KtoYList_Tran   = []
tVarList   = []
tVarBase = BaselineType.TranShkStd[0] ** 2
for tVar in points * tVarBase:
    tVarList.append(tVar) # Variance is std squared
    savRteList_Tran.append(calcSavRte(EstimationEconomy,"TranShkStd",[tVar ** 0.5]))
    KtoYList_Tran.append(0.25*np.mean(np.array(EstimationEconomy.KtoYnow_hist)[EstimationEconomy.ignore_periods:]))

# %% {"code_folding": [0]}
# Plot transitory variance versus saving measures
plotReg(tVarList,savRteList_Tran,
        xMin=tVarList[1]-0.001,xMax=tVarList[-1]+0.001,yMin=savRteList[1]-0.01,yMax=savRteList[-1]+0.01,
        xLbl=r'Variance of Transitory Shocks, $\sigma^{2}_{\theta}$',
        yLbl='Aggregate Saving Rate',
        Title='Uncertainty vs Saving',
        fileName='savRteVSTranShkVar'
       )
plt.show(block=False)
plotReg(tVarList,KtoYList_Tran,
        xMin=tVarList[1]-0.001,xMax=tVarList[-1]+0.001,yMin=savRteList[1]-0.01,yMax=KtoYList[-1]+0.1,
        xLbl=r'Variance of Permanent Shocks, $\sigma^{2}_{\psi}$',
        yLbl='Net Worth/Income',
        Title='Uncertainty vs Net Worth Ratio',
        fileName='BvsTranShkVar'
       )
plt.show(block=False)                

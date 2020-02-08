# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Tractable Buffer Stock Model

# %% [markdown]
# The [TractableBufferStock](http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/TractableBufferStock/) model is a (relatively) simple framework that captures all of the qualitative, and many of the quantitative features of optimal consumption in the presence of labor income uncertainty.  

# %% {"code_folding": [0]}
# This cell has just a bit of initial setup. You can click the arrow to expand it.
# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import HARK 
from time import clock
from copy import deepcopy
mystr = lambda number : "{:.3f}".format(number)

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from HARK.utilities import plotFuncs
from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType

# %% [markdown]
# The key assumption behind the model's tractability is that there is only a single, stark form of uncertainty.  So long as an employed consumer remains employed, his labor income will rise at a constant rate.  But between any period and the next there is constant hazard $p$ of transitioning to the "unemployed" state. Unemployment is irreversible, like retirement or disability.  When unemployed, the consumer receives a fixed amount of income (for simplicity, zero).
#
# \begin{eqnarray*}
# \mathbb{E}_{t}[V_{t+1}^{\bullet}(M_{t+1})] & = & (1-p)V_{t+1}^{e}(M_{t+1})+p V_{t+1}^{u}(M_{t+1})
# \end{eqnarray*}
#
# A consumer with CRRA utility $U(C) = \frac{C^{1-\rho}}{1-\rho}$ solves an optimization problem that looks standard (where $P$ is Permanent income and $\Gamma$ is the income growth factor):
#
# \begin{eqnarray*}
# V_t(M_t) &=& \max_{C_t} ~ U(C_t) + \beta \mathbb{E}[V_{t+1}^{\bullet}], \\
# M_{t+1} &=& R A_t + \mathbb{1}(P_{t+1}), \\
# P_{t+1} &=& \Gamma_{t+1} P_t,
# \end{eqnarray*}
#
# where $\mathbb{1}$ is an indicator of whether the consumer is employed in the next period.
#
# Under plausible parameter values the model has a target level of $\check{m} = M/P$ (market resources to permanent income) with an analytical solution that exhibits plausible relationships among all of the parameters.  (See the [linked handout](http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/Consumption/TractableBufferStock/) for details).

# %% {"code_folding": [0]}
# Define a parameter dictionary and representation of the agents for the tractable buffer stock model
TBS_dictionary =  {'UnempPrb' : .00625,    # Prob of becoming unemployed; working life of 1/UnempProb = 160 qtrs
                   'DiscFac' : 0.975,      # Intertemporal discount factor
                   'Rfree' : 1.01,         # Risk-free interest factor on assets
                   'PermGroFac' : 1.0025,  # Permanent income growth factor (uncompensated)
                   'CRRA' : 2.5}           # Coefficient of relative risk aversion
MyTBStype = TractableConsumerType(**TBS_dictionary)


# %% {"code_folding": [0]}
# Define a function that plots the employed consumption function and sustainable consumption function 
# for given parameter values

def makeTBSplot(DiscFac,CRRA,Rfree,PermGroFac,UnempPrb,mMax,mMin,cMin,cMax,plot_emp,plot_ret,plot_mSS,show_targ):
    MyTBStype.DiscFac = DiscFac
    MyTBStype.CRRA = CRRA
    MyTBStype.Rfree = Rfree
    MyTBStype.PermGroFac = PermGroFac
    MyTBStype.UnempPrb = UnempPrb
    
    try:
        MyTBStype.solve()
    except:
        print('Those parameter values violate a condition required for solution!')    
    
    plt.xlabel('Market resources $M_t$')
    plt.ylabel('Consumption $C_t$')
    plt.ylim([cMin,cMax])
    plt.xlim([mMin,mMax])
    
    m = np.linspace(mMin,mMax,num=100,endpoint=True)
    if plot_emp:
        c = MyTBStype.solution[0].cFunc(m)
        c[m==0.] = 0.
        plt.plot(m,c,'-b')
        
    if plot_mSS:
        plt.plot([mMin,mMax],[(MyTBStype.PermGroFacCmp/MyTBStype.Rfree + mMin*(1.0-MyTBStype.PermGroFacCmp/MyTBStype.Rfree)),(MyTBStype.PermGroFacCmp/MyTBStype.Rfree + mMax*(1.0-MyTBStype.PermGroFacCmp/MyTBStype.Rfree))],'--k')
        
    if plot_ret:
        c = MyTBStype.solution[0].cFunc_U(m)
        plt.plot(m,c,'-g')
    
    if show_targ:
        mTarg = MyTBStype.mTarg
        cTarg = MyTBStype.cTarg
        targ_label = '$m^* =$' + mystr(mTarg) + '\n$c^* =$' + mystr(cTarg)
        plt.annotate(targ_label,xy=(0.0,0.0),xytext=(0.8,0.05),textcoords='axes fraction')
    
    plt.show()
    return None

# Define widgets to control various aspects of the plot

# Define a slider for the discount factor
DiscFac_widget = widgets.FloatSlider(
    min=0.9,
    max=0.99,
    step=0.0002,
    value=0.95,
    continuous_update=False,
    readout_format='.4f',
    description='$\\beta$')

# Define a slider for relative risk aversion
CRRA_widget = widgets.FloatSlider(
    min=0.1,
    max=8.0,
    step=0.01,
    value=2.5,
    continuous_update=False,
    readout_format='.2f',
    description='$\\rho$')

# Define a slider for the interest factor
Rfree_widget = widgets.FloatSlider(
    min=1.00,
    max=1.04,
    step=0.0001,
    value=1.01,
    continuous_update=False,
    readout_format='.4f',
    description='$R$')


# Define a slider for permanent income growth
PermGroFac_widget = widgets.FloatSlider(
    min=0.98,
    max=1.02,
    step=0.0002,
    value=1.0025,
    continuous_update=False,
    readout_format='.4f',
    description='$\\Gamma$')

# Define a slider for unemployment (or retirement) probability
UnempPrb_widget = widgets.FloatSlider(
    min=0.00001,
    max=0.10,
    step=0.00001,
    value=0.00625,
    continuous_update=False,
    readout_format='.5f',
    description='$\\mho$')

# Define a text box for the lower bound of M_t
mMin_widget = widgets.FloatText(
    value=0.0,
    step=0.1,
    description='$M$ min',
    disabled=False)

# Define a text box for the upper bound of M_t
mMax_widget = widgets.FloatText(
    value=50.0,
    step=0.1,
    description='$M$ max',
    disabled=False)

# Define a text box for the lower bound of C_t
cMin_widget = widgets.FloatText(
    value=0.0,
    step=0.1,
    description='$C$ min',
    disabled=False)

# Define a text box for the upper bound of C_t
cMax_widget = widgets.FloatText(
    value=1.5,
    step=0.1,
    description='$C$ max',
    disabled=False)

# Define a check box for whether to plot the employed consumption function
plot_emp_widget = widgets.Checkbox(
    value=True,
    description='Plot employed $C$ function',
    disabled=False)

# Define a check box for whether to plot the retired consumption function
plot_ret_widget = widgets.Checkbox(
    value=True,
    description='Plot retired $C$ function',
    disabled=False)

# Define a check box for whether to plot the sustainable consumption line
plot_mSS_widget = widgets.Checkbox(
    value=True,
    description='Plot sustainable $C$ line',
    disabled=False)

# Define a check box for whether to show the target annotation
show_targ_widget = widgets.Checkbox(
    value=True,
    description = 'Show target $(m,c)$',
    disabled = False)


# %% [markdown]
# ## Target Wealth
#
# Whether the model exhibits a "target" or "stable" level of the wealth-to-permanent-income ratio for employed consumers depends on whether the 'Growth Impatience Condition' (the GIC) holds:
#
# \begin{align}\label{eq:GIC}
#  \left(\frac{(R \beta (1-\mho))^{1/\rho}}{\Gamma}\right)  & <  1
# \\ \left(\frac{(R \beta (1-\mho))^{1/\rho}}{G (1-\mho)}\right)  &<  1
# \\ \left(\frac{(R \beta)^{1/\rho}}{G} (1-\mho)^{-\rho}\right)  &<  1
# \end{align}
# and recall (from [PerfForesightCRRA](http://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/)) that the perfect foresight 'Growth Impatience Factor' is 
# \begin{align}\label{eq:PFGIC}
# \left(\frac{(R \beta)^{1/\rho}}{G}\right)  &<  1
# \end{align}
# so since $\mho > 0$, uncertainty makes it harder to be 'impatient.'
#
# Think of someone who, in the perfect foresight model, was 'poised': Exactly on the knife edge between patience and impatience.  Now add a precautionary saving motive; that person will now (to some degree) be pushed off the knife edge in the direction of 'patience.'  So, in the presence of uncertainty, the conditions on parameters other than $\mho$ must be stronger in order to guarantee 'impatience' in the sense of wanting to spend enough for your wealth to decline _despite_ the extra precautionary motive.

# %% {"code_folding": [0, 3]}
# Make an interactive plot of the tractable buffer stock solution

# To make some of the widgets not appear, replace X_widget with fixed(desired_fixed_value) in the arguments below.
interact(makeTBSplot,
         DiscFac = DiscFac_widget,
         CRRA = CRRA_widget,
#         CRRA = fixed(2.5),
         Rfree = Rfree_widget,
         PermGroFac = PermGroFac_widget,
         UnempPrb = UnempPrb_widget,
         mMin = mMin_widget,
         mMax = mMax_widget,
         cMin = cMin_widget,
         cMax = cMax_widget,
         show_targ = show_targ_widget,
         plot_emp = plot_emp_widget,
         plot_ret = plot_ret_widget,
         plot_mSS = plot_mSS_widget,
        );



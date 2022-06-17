#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bentPlumeAnalyser import *
from fumarolePlumeModel import *
from scipy.io.matlab import loadmat
from scipy.integrate import ode, solve_ivp
from itertools import product
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import json

# Set numpy options, notably for the printing of floating point numbers
np.set_printoptions(precision=6)

# Set matplotlib options
mpl.rcParams['figure.dpi'] = 300

#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[41]:


def integrator(V0, p):
    # Initialise an integrator object
    r = ode(derivs).set_integrator('lsoda', nsteps=1e6)
    r.set_initial_value(V0, 0.)
    r.set_f_params(p)
    
    # Define state vector and axial distance
    V = []    # State vector
    s = []    # Axial distance
    V.append(V0)
    s.append(sexp[0])
    
    # Define the individual variables - these will be calculated at run time
    Q, M, F, theta = 0., 0., 0., 0.
    Q = np.float64(Q0)
    M = np.float64(M0)
    F = np.float64(F0)
    theta = np.float64(theta0)
    
    ####################################

    # Integrate, whilst successful, until the domain size is reached
    ind = 0
    while r.successful() and r.t < t1 and M >= 0. and ind < len(dsexp):
        dt = dsexp[ind]
        r.integrate(r.t + dt)
        V.append(r.y)
        s.append(r.t)
        Q, M, F, theta = r.y
        ind += 1
    s = np.array(s)
    V = np.float64(np.array(V))
    return s, V


def integrator(V0, p, x=None):
    if x is None:
        x   = np.linspace(0, 25, 21)
    sol = solve_ivp(derivs, [x[0], x[-1]], V0, args=(p,), t_eval=x)
    return sol.t, sol.y.T


# In[25]:


run = 3

# Import table of experimental conditions
GCTA = pandas.read_excel('./data/ExpPlumes_for_Dai/TableA1.xlsx', sheet_name='CGSdata', skiprows=2, 
                         names=('exptNo', 'rhoa0', 'sig_rhoa0', 'N', 'sig_N', 'rho0', 'sig_rho0', 'U0', 'sig_U0', 'W', 'sig_W', 
                                'gp', 'sig_gp', 'Q0','sig_Q0', 'M0', 'sig_M0', 'F0', 'sig_F0', 'Ri_0', 'sig_Ri_o', 'W*','sig_W*'))

# Extract densities of ambient and plume, and calculate g' at the source
expt  = GCTA[GCTA['exptNo'] == run]
rhoa0 = expt['rhoa0']
rho0  = expt['rho0']
g = 981 #cm/s²
gp0   = (rhoa0 - rho0) / rhoa0 * g

parameters = pandas.read_excel('./data/ExpPlumes_for_Dai/TableA1.xlsx', sheet_name='CGSparameters')
b0theoretical = parameters[parameters['property'] == 'nozzleSize']['value'].values[0]
u0theoretical = expt['U0'].values[0]


# ### Create a synthetic dataset for a vertical plume

# In[26]:


exptNo      = 3
plotResults = True

V0, p = loadICsParameters(pathname, exptNo, alpha=0.09, beta=0, m=1)

p = list(p)
p[1], p[4] = 0., 0.
p = tuple(p)

t1 = 30.
dt = .1
sexp = np.arange(0., t1 + dt, dt)
dsexp = np.diff(sexp)


# In[43]:


Q0, M0, F0, theta0 = V0
s, V = integrator(V0, p)

sexp = np.copy(sexp)

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(V, s, '-')

Q, M, F, theta = [V[:,i] for i in range(4)]
b, u, gp = Q / np.sqrt(M), M / Q, F / Q

V2 = np.array([b, u, gp]).T
ax[1].plot(V2, s, '-')
ax[1].set_xlim((-5, 30))
ax[1].grid()

sexp = np.copy(s)
dsexp = np.diff(sexp)


# ### Add noise to signal

# In[44]:


plt.plot(V2, s, '-')
V3 = V2.copy()
for i in range(3):
    noise = np.random.normal(0, .3, len(sexp))
    V3[:,i] = V2[:,i] + noise
    plt.plot(V3[:,i], s, '.', c='C%d' % i, ms=1.5)
plt.xlim((-5, 30))

bexp, uexp, gpexp = [V3[:,i] for i in range(3)]

print(len(bexp))


# In[49]:


gp0 = V0[2] / V0[0]


# In[50]:


#p = (0.05, .5, .012, 2., 4.)
nGrid = 21   # Number of grid points
b0Vec = np.linspace(.05, 2, nGrid) #cm
u0Vec = np.linspace(5, 30, nGrid) #cm/s
Q0Vec = u0Vec * b0Vec**2 #cm3/s
M0Vec = Q0Vec * u0Vec #cm4/s2

theta0 = np.pi / 2

objFn, initialConds = [], []

sequence = [Q0Vec, M0Vec]

for (Q0, M0) in list(product(*sequence)):
    F0 = Q0 * gp0
    V0 = [Q0, M0, F0, theta0]
    
    # Call the 'integrator' function (defined above) to solve
    # the model
    s, V = integrator(V0, p, x=sexp)
    Q, M, F, theta = [V[:,i] for i in range(4)]
    
    #######################################
    
    b  = Q / np.sqrt(M) # Factor of sqrt{2} to correspond with top-hat model
    u  = M / Q
    gp = F / Q
    
    Vexp = np.array([bexp]).ravel(order='C')
    Vsyn = np.array([b]).ravel(order='C')
    Vexp = Vexp[:len(Vsyn)]

    objFn.append(objectiveFn(Vexp[:-1], Vsyn[:-1], p=p))
    initialConds.append(V0)
#     ax.plot(b, s, '-', label='Model %.4f %.4f %.4f' % (Q0, M0, F0))
    
#ax.legend(loc=5)
# ax.set_xlim((0, 25))

# Transform initialConds and objFn from lists to arrays, 
# reshaping the latter 
initialConds = np.array(initialConds)
objFn_b = np.array(objFn).reshape((nGrid, nGrid))


# In[47]:


# objectiveFn(Vexp[:-1], Vsyn[:-1], p=p)

print(len(Vsyn))
print(len(Vexp))


# In[48]:


objFn, initialConds = [], []

sequence = [Q0Vec, M0Vec]

for (Q0, M0) in list(product(*sequence)):
    F0 = Q0 * gp0
    V0 = [Q0, M0, F0, theta0]
    
    # Call the 'integrator' function (defined above) to solve
    # the model
    s, V = integrator(V0, p, x=sexp)
    Q, M, F, theta = [V[:,i] for i in range(4)]
    
    #######################################
    
    b  = Q / np.sqrt(M) # Factor of sqrt{2} to correspond with top-hat model
    u  = M / Q
    gp = F / Q
    
    Vexp = np.array([gpexp]).ravel(order='C')
    Vsyn = np.array([gp]).ravel(order='C')
    Vexp = Vexp[:len(Vsyn)]

    objFn.append(objectiveFn(Vexp[:-1], Vsyn[:-1], p=p))
    initialConds.append(V0)
#     ax.plot(b, s, '-', label='Model %.4f %.4f %.4f' % (Q0, M0, F0))
    
#ax.legend(loc=5)
# ax.set_xlim((0, 25))

# Transform initialConds and objFn from lists to arrays, 
# reshaping the latter 
initialConds = np.array(initialConds)
objFn_gp = np.array(objFn).reshape((nGrid, nGrid))


# In[39]:


fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True)


## Solution for b 
ax[0].pcolor(b0Vec, u0Vec, np.log10(objFn_b))

## Optimal values
ui, bi = np.where(objFn_b == objFn_b.max())
bOpt = b0Vec[bi[0]]
uOpt = u0Vec[ui[0]]

ax[0].plot(bOpt, uOpt, 'r.', ms=8, label='Values with max. prob.')
hp, = ax[0].plot(b0theoretical, u0theoretical, 'w.', ms=8, label='Supposed values')
hp.set_markeredgecolor('k')
ax[0].set_title('Using $b$ as observable')
ax[0].set_xlabel('b0')
ax[0].set_ylabel('u0')

## Solution for gp 
ax[1].pcolor(b0Vec, u0Vec, np.log10(objFn_gp))

## Optimal values
ui, bi = np.where(objFn_gp == objFn_gp.max())
bOpt = b0Vec[bi[0]]
uOpt = u0Vec[ui[0]]


ax[1].plot(bOpt, uOpt, 'r.', ms=8, label='Values with max. prob.')
hp, = ax[1].plot(b0theoretical, u0theoretical, 'w.', ms=8, label='Supposed values')
hp.set_markeredgecolor('k')
ax[1].set_title('Using $g\'$ as observable')
ax[1].set_xlabel('b0')

fig.savefig('syntheticPlumeObjectiveFunction.png')


# In[19]:


max(objFn)


# In[ ]:





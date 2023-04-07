#####################################
######## Model 4 adding the heat diffusion equation #############
##################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.integrate import quad
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import pdb
from pdeSolver import *
######### Constants ######################
e = 1.6E-19
m = 9.31E-31
kB = 1.38E-23# J/K
#############################################
##### device constants ############
###################################################
l = 10E-6 # m length of device
w = 100E-9# m width of device
d = 5E-9#cm thickness of device
N0 = 2E21# cm-3 density of all carriers
F = 10E-6# J/cm2 Laser fluence
toub = 1E-10# s cooper pair breaking time
tour = 0.6E-12#s Quasi-particle recombination rate
toues = 3.5E-12#s Phonon escape time
Tc = 4# K for WSi
delt = 50E-7#cm laser penetration depth
Rf = 0.1 # reflection coefficient of YBCO
tstep = 0.1E-12
tend = 100E-12
rho0 = 2.4E-6# resistivity in units of rho/m
alpha = 3E-7 ## Very poor approximation: https://journals.aps.org/pr/pdf/10.1103/PhysRev.181.1127
mu = 8E5 # interface thermal conductivity in W/m2K
Tsub = 2
L = 2.45E-8 # Lorentz number Womega/K2
############# Setting up trial #########
#J = 0
k  = 0.02
mu = 0
rho0 =350*d# Ohm cm 
jc0 = 8E9#A/m2
a=58.5E-3# mK
b=194#mK
c=1263#mK
Ce0 = 134.7 #eVcm-3K -  134.7 J/m3K Kg/cms2k2
J = 0.9*jc0 #### Device current 
#s_e_ph = 6.180
s_e_ph = 1E9
########################################
#####################################################
#####################################################
##### calculations ############
T = 2# K
Nqpt =N0*(T/Tc)**2
Rec = 2/(tour*Nqpt)
Nwt = toub*Nqpt/tour
sig = 1E-12# s pulse width
t0 = 5E-10
fr = 80E6 # Hz laser frequency
lp = 1E3/(np.pi*1E-3)**2# mW/cm2 laser power from meter
pE = lp/fr # J/cm2 Laser fluence, pulse energy
lI = pE/sig # peak laser power  
def gen(t):
    return 1E14*lI*(1-Rf)*(1 - np.exp(-d/delt))*np.exp(-(t - t0)**2/(2*sig**2))/d # Jcm-2
ngen = quad(gen,-100E-12,tend)
def eqns(Y,t):
    dY = np.zeros(2)
    dY[0] = gen(t) - Rec*(Y[0])**2 + 2*Y[1]/toub # excess quasi particles
    dY[1] = 0.5*Rec*(Y[0])**2 - Y[1]/toub - Y[1]/toues # excess holes
    return dY
Y_int=np.array([0,0])
tspan = np.arange(0,tend,tstep)
res = odeint(eqns,Y_int,tspan)
delE = np.append(np.diff(res[:,0])/np.diff(tspan),0)
delP = np.append(np.diff(res[:,1])/np.diff(tspan),0)
#pdb.set_trace()
fig1,ax1 = plt.subplots()
ax1.plot(tspan[5:],res[5:,1],'r')
#ax1.plot(tspan,gen(tspan),'r')
ax1.set_ylabel('Phonon population')
ax2=ax1.twinx()
ax2.plot(tspan[5:],res[5:,0],'b')
ax2.set_ylabel('Super electrons')
fig1.savefig('polulationDynamics.png')
plt.close(fig1)
#plt.show()
def Tsw(j):
    return Tc*np.real((1 - (j/jc0)**(2/3))**(1/2))
Del = lambda u: 1.76*kB*Tc*np.tanh(np.pi*np.sqrt(2*1.43*(Tc/u - 1)/3)/1.76)
def getCe(u):
    temp = np.zeros(len(u))
    for i in range(len(u)):
        if u[i] < Tc:
            temp[i] = Ce0*Tc*2.43*np.exp(-Del(u[i])/(kB*u[i]))
        else:
            temp[i] =  Ce0*u[i]
    return temp
def getKe(u):
    temp = np.zeros(len(u))
    for i in range(len(u)):
        if u[i] < Tc:
            temp[i]= L*u[i]**2/(rho0*Tc)
        else:
            temp[i] =  L*Tc/rho0
    return temp
def getRho(j,u):
    return rho0*(0.5 + 0.5*np.tanh(((u - Tsw(j))/a)*(0.5 - 0.5*np.tanh((u - Tsw(j))/b)) + ((u - Tsw(j))/c)*(0.5 - 0.5*np.tanh((u - Tsw(j))/b))))
t = np.linspace(2,Tc,100)
jt = np.linspace(0,jc0,100)
rhoc = np.zeros([len(t),len(jt)])
for i in range(len(t)):
    for j in range(len(jt)):
        rhoc[i,j] = getRho(jt[j],t[i])
fig1,ax1 = plt.subplots(2,2)
ax1[0,0].plot(jt,Tsw(jt))
ax1[0,0].set_title('Switching temperature')
ax1[0,1].plot(t,getCe(t))
ax1[0,1].set_title('Heat Capacity')
ax1[1,0].plot(t,getKe(t)/getCe(t))
ax1[1,0].set_title('thermal conductivity/heat capacity')
im=ax1[1,1].imshow(rhoc,cmap='coolwarm')
ax1[1,1].set_title('Rho(j,T)')
ax1[1,1].set_ylabel('Temperature')
ax1[1,1].set_xlabel('Current Density')
fig1.colorbar(im)
fig1.savefig('parameters.png')
plt.close(fig1)



##########
## Lets divide the length into some number of units N
#####################
nS = 150 # number of spatial units
xGrid = np.linspace(0,l,nS)
dx = xGrid[1] - xGrid[0]
###########################
###### We are looking at time dynamics. So time step as dT
dt = 1*dx**2*getCe([T])/getKe([T])
print(dt)
#dt = 1E-11
nT = 360
tGrid = np.linspace(0,nT*dt,nT)
k1 = L * T/rho0
beta = (mu/d - J**2*alpha)
gamma = J**2*rho0 + mu*Tsub/d
UMat = np.zeros([nT,nS])
UMat[0,:] = [T for j in range(nS)] # initial condition
initial = np.ones(nS)*Tsub

def getSource(u,t):
    sv = np.zeros(len(u))
    sv = (w*d*dx*getRho(J,u)*J**2) # return joule heating 
    #sv = (w*dx*getRho(J,u)*J**2) # return joule heating 
    #sv[5] = sv[5] + gen(t)*1E-26
    return sv

def getLosses(u,t):
    lv = np.zeros(len(u))
    lv = s_e_ph*(u - Tsub)
    return lv
def NML(u):
    return -1*(u - T0)
def NMR(u):
    return -1*(u - T0)
Bounds = {
        'LT':'D',
        'LV':Tsub,
        'RT':"D",
        'RV':Tsub
        }
def C(u_in):
    #return getKe(u_in)/getCe(u_in)
    return [1E-3 for i in range(len(u_in))]
def S(X,t,u_in):
    x0 = l/3
    sig = l/50
    t0 = 15E-10
    sigt =2E-10
    sv = 1E13*np.exp(-(X - x0)**2/(2*sig**2))*np.exp(-(t-t0)**2/(2*sigt**2))/getCe(u_in)
    sv =sv +  getSource(u_in,t) - getLosses(u_in,t)
    return sv
    #return [0 for i in range(len(u))]
fig2,ax2 = plt.subplots()
sValue=np.zeros([nS,nT])
for t in range(len(tGrid)):
    sValue[:,t] = S(xGrid,tGrid[t],initial)
im=ax2.imshow(sValue)
fig2.colorbar(im)
fig2.savefig('source.png')
plt.close(fig2)

sys = Solver1D(xGrid,initial,tGrid,Bounds,C,S)
soln = sys.solve()
sys.plotSolution()

##########################################################################
##################### Finite difference solver programs for 1D and 2S systems #####
##################### Works well as of 5th April 2023. 
############################### Version 1###################
##########################################################################
#### Version 2 incorporating multiple systems solution simultaneously
##########################################################################
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import pdb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
class Solver1D:
    ############# initialize the class instance: 
    ##### mesh - created by numpy meshgrid. For 1D, meshgrid should take a array containing mesh points
    ##### tGrid - array of time stamps on which the solution is required.
    #### Nsys - number of simultaneous equations to be solved. All further parameters should be arrays of size Nsys
    #### initial[Nsys, meshSize] an array of [Nsys,meshSize] for 1D containing the initial value of the solutions for each dependent variable
    #### Bounds[Nsys] - Boundary conditions array. Each element in the array is a dictionary. It should contain the following elements
    ################# Bounds[i]["LT"] - type of left boundary. Currently it can take 'D' for Dirichlet and 'N' for Neumann
    ################# Bounds[i]["LV"] - Value of left boundary. The argument should be a function which takes the solver class instance as a variable.
    ################# Bounds[i]["RV"] and ["RT"] are the same for the right boundary
    #### Cfun - a function which returns the coefficient of the gradient term in the PDE. It takes as an argument the 'x' values at which the coefficient is required. It also gets the isntance of the solver.
    #### Sfun - a function which returns the source/forcing term in the PDE. It takes as an argument the 'x' values at which the source value is required. It also gets the isntance of the solver.
    def __init__(self,mesh,tGrid,Nsys,initial,Bounds,Cfun,Sfun):
        self.mesh = mesh[0] # number of spatial units
        self.Nsys=Nsys
        self.dx = self.mesh[1] - self.mesh[0]
        self.dt = tGrid[1] - tGrid[0]
        self.nT = len(tGrid)
        self.nS = len(self.mesh)
        self.tGrid = tGrid
        self.UMat = np.zeros([Nsys,self.nT,self.nS])
        self.Boundary = Bounds
        self.C = Cfun 
        self.S = Sfun
        self.UMat[:,0,:] = initial # initial condition
        self.tSol = 0## variable having the last solved time
    def solve(self):
        tGrid = np.linspace(0,self.nT*self.dt,self.nT)
        for i in range(self.nT-1): ### for each time step
            self.tSol=i
            for m in range(self.Nsys): #### for every system
                sigma = np.multiply(self.C[m](self.mesh,self),self.dt/(2*self.dx*self.dx))
                AMat = diags([-sigma, 1 + 2*sigma, -sigma],[-1,0,1],shape=(self.nS,self.nS),format = 'csr')
                BMat = diags([sigma, 1 - 2*sigma, sigma],[-1,0,1],shape=(self.nS,self.nS), format = 'csr')
                CMat = BMat.dot(self.UMat[m,i,:]) + np.multiply(self.dt,self.S[m](self.mesh,tGrid[i],self))
                #### setting the left conditions #####
                if self.Boundary[m]["LT"] == 'D':
                    AMat[0,0] = 1
                    AMat[0,1] = 0
                    CMat[0] = self.Boundary[m]["LV"]
                elif self.Boundary["LT"] == 'N':
                    AMat[0,0] = 1
                    AMat[0,1] = -1
                    CMat[0] = self.Boundary[m]["LV"](self.UMat[m,i,0])
                #### setting the right conditions #####
                if self.Boundary[m]["RT"] == 'D':
                    AMat[-1,-1] = 1
                    AMat[-1,-2] = 0
                    CMat[-1] = self.Boundary[m]["RV"]
                elif self.Boundary[m]["RT"] == 'N':
                    AMat[-1,-1] = 1
                    AMat[-1,-2] = -1
                    CMat[-1] = self.Boundary[m]["RV"](self.UMat[m,i,0])
                self.UMat[m,i+1,:] = spsolve(AMat,CMat)#/getCe(UMat[i,:])
            self.sol = {"res":self.UMat,"x":self.mesh,"t":tGrid}
        return self.UMat
    def plotSolution(self):
        for m in range(self.Nsys):
            fig2, ax2 = plt.subplots()
            im = ax2.pcolormesh(self.mesh*1E6,self.tGrid*1E9,self.UMat[m],cmap='seismic')
            ax2.set_xlabel('position (um)')
            ax2.set_ylabel('time (ns)')
            fig2.colorbar(im)
            fig2.savefig('2Dsolution' + '-sys-'+str(m)+'.png')
            plt.close(fig2)
            return fig2
        ### uncommenmt below if you want a video
        #fig2, ax2 = plt.subplots()
        #def animate(i):
        #    ax2.clear()
        #    line=ax2.plot(self.mesh,self.UMat[i,:])
        #    return line
        #anim = FuncAnimation(fig2,func=animate,frames=self.nT,interval=1,repeat=True,blit=True)
        #writer = PillowWriter(fps=30)
        #anim.save("solution1D.gif",dpi=300, writer=writer)



########################################
####### This is implemented using factored approximations
########################################

class Solver2D:
    def __init__(self,mesh,tGrid,Nsys,initial,Bounds,Cfun,Sfun):
        self.mesh = mesh # number of spatial units
        self.Nsys = Nsys
        self.nT = len(tGrid)
        self.Boundary = Bounds
        self.C = Cfun 
        self.S = Sfun
        self.xPos = self.mesh[0][0,:]
        self.yPos = self.mesh[1][:,0]
        self.nSx = len(self.xPos)
        self.nSy = len(self.yPos)
        self.dx = self.mesh[0][0,1] - self.mesh[0][0,0]
        self.dy = self.mesh[1][1,0] -self.mesh[0][0,0]
        #############################################################
        ###### We are looking at time dynamics. So time step as dT
        ############################################################
        sys.UMat = np.zeros([Nsys,self.nT,self.nSx,self.nSy])
        self.dt =tGrid[1] - tGrid[0]
        self.UMat[:,0,:,:] = initial
        print(self.dx,self.dy,self.dt)
        self.tGrid = tGrid
        self.tSol = 0 ### variable holding the last solved time

    def setBC(self,BC,t,j,m,BD):
        if BD == 'X': ### boundary along the X axis - solutions are found along the Y-Axis
            X = self.yPos
            sigma_y = np.multiply(self.C[m]([self.xPos[j]],self.yPos,self),self.dt/(2*self.dy*self.dy))
            BCMat = diags([-sigma_y, 1 + 2*sigma_y, -sigma_y],[-1,0,1],shape=(self.nSy,self.nSy),format = 'csr')
        else:
            X = self.xPos
            sigma_x = np.multiply(self.C[m](self.xPos,[self.yPos[j]],self),self.dt/(2*self.dx*self.dx))
            BCMat = diags([-sigma_x, 1 + 2*sigma_x, -sigma_x],[-1,0,1],shape=(self.nSx,self.nSx),format = 'csr')
        #### setting the dirchilet left conditions #####
        if BC["LT"] == 'D':
            self.AMat[0,0] = 1
            self.AMat[0,1] = 0
            self.cMat[0] =BCMat.dot(BC["LV"](X,t))[j]
        elif BC["LT"] == 'N':
            self.AMat[0,0] = 1
            self.AMat[0,1] = -1
            self.cMat[0] = BCMat.dot(BC["LV"](X,t))[j]
            
        #### setting the dirchilet right conditions #####
        if BC["RT"] == 'D':
            self.AMat[-1,-1] = 1
            self.AMat[-1,-2] = 0
            self.cMat[-1] =BCMat.dot(BC["RV"](X,t))[j]
        elif BC["RT"] == 'N':
            self.AMat[-1,-1] = 1
            self.AMat[-1,-2] = -1
            self.cMat[-1] =BCMat.dot(BC["RV"](X,t))[j]
    
    def solve(self):
        ### the solution is done in two steps (factorization) ##########
        for i in range(self.nT - 1): ### we always calculate t+1. So its total time step for -1
            self.tSol=i
            for m in range(self.Nsys):

                eta = np.zeros([self.nSx,self.nSy])
                gamma = np.zeros([self.nSx,self.nSy])
                ##### Get the RHS ##########
                for j in range(self.nSy):
                    sigma_x = np.multiply(self.C[m](self.xPos,[self.yPos[j]],self),self.dt/(2*self.dx*self.dx))
                    self.BMatx = diags([sigma_x, 1 - 2*sigma_x, sigma_x],[-1,0,1],shape=(self.nSx,self.nSx), format = 'csr')
                    eta[:,j] = self.BMatx.dot(self.UMat[m,i,:,j])
                for j in range(self.nSx):
                    sigma_y = np.multiply(self.C[m]([self.xPos[j]],self.yPos,self),self.dt/(2*self.dy*self.dy))
                    self.BMaty = diags([sigma_y, 1 - 2*sigma_y, sigma_y],[-1,0,1],shape=(self.nSy,self.nSy), format = 'csr')
                    self.cMat = self.BMaty.dot(eta[j,:]) + np.multiply(self.dt,self.S([self.mesh[0][0,j]],self.mesh[1][:,0],self.tGrid[i])[0])
                    ##### Get the first factor properties############
                    self.AMat = diags([-sigma_x, 1 + 2*sigma_x, -sigma_x],[-1,0,1],shape=(self.nSy,self.nSy),format = 'csr')
                    #### boundary conditions for the first factor
                    BC = {
                            "LT":self.Boundary["LT"],
                            "LV":self.Boundary["LV"],
                            "RT":self.Boundary["RT"],
                            "RV":self.Boundary["RV"]
                            }
                    self.setBC(BC,t,j,'Y')## sets the appropriate BC, if along Y, send two arrays of suze Nx
                    gamma[j,:] = spsolve(self.AMat,self.cMat.T) # solve the first factor
                    ##### Get the second factor properties############
                for j in range(self.nSy):
                    self.AMat = diags([-sigma_x, 1 + 2*sigma_x, -sigma_x],[-1,0,1],shape=(self.nSx,self.nSx),format = 'csr')
                    #### boundary conditions for the second factor
                    BC = {
                            "LT":self.Boundary["BT"],
                            "LV":self.Boundary["BV"],
                            "RT":self.Boundary["TT"],
                            "RV":self.Boundary["TV"]
                            }
                    self.setBC(BC,t,j,'X')## sets the appropriate BC, if along X, send two arrays of size Ny
                    self.UMat[m,i+1,:,j] = spsolve(self.AMat,gamma[:,j]) # solve the first factor
        return self.UMat

    def plotSolution(self):
        fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
        def animate(i):
            ax2.cla()
            ax2.plot_surface(mesh[0],mesh[1],UMat[i,:,:].T)
        return fig2
        anim = FuncAnimation(fig2,func=animate,frames=nT,interval=1,repeat=False)
        writer = PillowWriter(fps=30)
        anima.save("solution.mp4", writer=writer)



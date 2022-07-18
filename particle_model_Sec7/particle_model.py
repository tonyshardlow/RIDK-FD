#
import numpy as np
#
import scipy
from scipy import sparse
from scipy.spatial import KDTree
#
import sympy
from sympy.utilities.lambdify import lambdify
#
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
##############
# set-up potential
#
def get_potential(params_m):
    x=sympy.Symbol('x')
    y=sympy.Symbol('y')
    Vext=params_m["external"]
    print("External potential ", Vext)
    Vext_x = sympy.diff(Vext, "x")
    Vext_y = sympy.diff(Vext, "y")
    # grad
    func_numpy_x = lambdify((x, y), Vext_x, modules=[np], dummify=False)
    #
    func_numpy_y = lambdify((x, y), Vext_y, modules=[np], dummify=False)
    def ext_grad(X):
            return np.array(
                [func_numpy_x(X[:, 0], X[:, 1]), func_numpy_y(X[:, 0], X[:, 1])]
            ).T
    params_m["ext_grad"]=ext_grad
##################
# initialise q,p,state with 
# N1,N2 particles of type 1,2
# on periodic of width L
def init_particles(params_m):
  # number of particles of each species
  N0=int(params_m["N0"])
  N1=int(params_m["N1"])
  N=int(N0+N1)
  params_m["N"]=N
  print("Creating",N0,N1," particles")
  # domain length (assume squre)
  L=params_m["Ly"]
  # iniital momentum
  p=np.zeros((N,2))
  # initial position
  if "random_init" in params_m:
    q=np.random.rand(N,2)*params_m["Ly"]
  else:
    q=np.empty((N,2))
    q[:N0,:]=params_m["mu0"]+params_m["sig0"]*np.random.randn(N0,2) 
    q[N0:,:]=params_m["mu1"]+params_m["sig1"]*np.random.randn(N1,2) 
  # project onto periodic domain
  q=q % params_m["Ly"] # 
  # initialise states
  state=np.zeros(N, dtype=int)
  state[N0:]=1 # state 1 to partcles from N0...N
  # set-up external potential
  get_potential(params_m)
  #
  return q,p,state
#########
# one time step
def onestep(q,p,dt,params_m):
  # get number of particles 
  N=q.shape[0] # from number of rows of q
  ## update q 
  q_=q+dt*p
  ## compute gradients
  q_=q_ % params_m["Ly"] # periodic domain
  dVq=params_m["ext_grad"](q_)
  ## compute noise
  xi=np.random.randn(N,2)
  # update p
  ## (1+gamm dt)p= p-grad V dt + sigma sqrt(dt) noise
  p_=(p-dVq *dt + params_m["sigma"]*xi*np.sqrt(dt))/(1+params_m["gamma"]*dt)
  return q_,p_

#############################
## React-step
## (0,1)->(1,1) with rate reaction_rate
def reactstep(q,state,dt,params_m):
  # get number of particles 
  N=q.shape[0] # from number of rows of q
  # threshold for reaction
  rate=params_m["reaction_rate"]
  # react or change state with probability
  threshold=1-np.exp(-rate*dt)
  # epsilon (radius) parameter
  eps=params_m["reaction_radius"]
  # convert positions to ND-tree
  # select state 0
  state0=np.argwhere(state==0).flatten()
  X0=q[state0,:]
  # select state 0
  state1=np.argwhere(state==1).flatten()
  X1=q[state1,:]
  #  reactions
  state=help_reactstep(X0,X1,state,state0,eps,threshold)
  # four repeats on periodic extensions
  state=help_reactstep(X0,X1-np.array([params_m["Lx"],0]),state,state0,eps,threshold)
  state=help_reactstep(X0,X1-np.array([0,params_m["Ly"]]),state,state0,eps,threshold)
  state=help_reactstep(X0,X1+np.array([params_m["Lx"],0]),state,state0,eps,threshold)
  state=help_reactstep(X0,X1+np.array([0,params_m["Ly"]]),state,state0,eps,threshold)
  # return
  return state
####
# helper function for above
def help_reactstep(X0,X1,state,state0,eps,threshold):
  kd_tree0 = KDTree(X0)
  kd_tree1 = KDTree(X1)
  # select pairs epsilon close
  sdm = kd_tree1.sparse_distance_matrix(kd_tree0, eps)
  # select lower triangular, convert format
  sdm_coo=scipy.sparse.tril(sdm,k=-1,format="coo")
  # toss coin on interactions and update state
  coin_toss=np.argwhere(np.random.rand(sdm_coo.nnz)<threshold )
  # change states
  if coin_toss.size>0:
    state[state0[sdm_coo.col[coin_toss]]]=1
  return state
#####################################
# reconstruct field
def construct_rho(state,q,params_m,params_n):
  xv=params_n["xv"]
  yv=params_n["yv"]
  # W_epslion
  def we(x,y): return (1/(2*np.pi*params_m["epsilon"]**2))*np.exp(-(np.minimum(x,params_m["Lx"]-x)**2+(np.minimum(y, params_m["Ly"]-y))**2)/(2*params_m["epsilon"]**2))
  # data strucutre to compute rho0, rho1
  val0=np.zeros(xv.shape)
  val1=np.zeros(xv.shape)
  # select for state
  state0=np.argwhere(state==0).flatten()
  state1=np.argwhere(state==1).flatten()
  #
  for k in state0:  
    val0=val0+we(xv-q[k,0],yv-q[k,1])
  for k in state1:  
    val1=val1+we(xv-q[k,0],yv-q[k,1])
  #
  scale=params_m["mass"]/params_m["N"]
  
  print("min ", np.min(scale*val0), np.min(scale*val1))
  print("max ", np.max(scale*val0), np.max(scale*val1))
  return scale*val0, scale*val1
##########
# 
def do_plot(state,q,params_m,a):
  # get fields rho0, rho1
  rho0,rho1=construct_rho(state,q,params_m,a)
  # 

  cmap0 = mpl.cm.get_cmap("Greens").copy()
  cmap1 = mpl.cm.get_cmap("Blues").copy()
  cmap0=cmap1
  #
  surf0 = a["ax"][0].contourf(a["xv"], a["yv"], rho0, a["levels0"], cmap=cmap0,extend='max')
  surf = a["ax"][1].contourf(a["xv"], a["yv"], rho1, a["levels1"], cmap=cmap1,extend='max')
  #
  a["ax"][0].set_aspect("equal")
  a["ax"][1].set_aspect("equal")
  #
  if "do_title" in a:
      a["ax"][0].set_title("Particles")  
      a["ax"][1].set_title(a["title"])  
  
  a["ax"][0].set_xlabel(r'$x$')
  a["ax"][1].set_xlabel(r'$x$')
  a["ax"][0].set_ylabel(r'$y$')
  #a["ax"][0].axis('off')
  #a["ax"][1].axis('off')
  return surf
##############################
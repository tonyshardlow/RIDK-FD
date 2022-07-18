#
# Main driver routine for particle reaction/diffusion model
# in Section 7
# 

import importlib
import pickle
#
import numpy as np
import scipy
#
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
#
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
# frame rate for gif
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation 
writer = PillowWriter(fps=20)
#
import particle_model
importlib.reload(particle_model)
##
params_m = {  # model parameters
    "gamma": 0.3, # dissipation
    "N0": 4.5e3, # number of particles
    "N1": 5e2,
    "mass":1,
    "sigma": 0.2, # noise coeff
    "reaction_rate":0.2, # reaction rate
    "Lx": 2.0 * np.pi, # domain size
    "Ly": 2.0 * np.pi,
    "reaction_radius": 0.15, # reaction radius
    "epsilon": 0.2, # to reconstuct empirical density
    "final_time": 25,
    # initial distribution
    "mu0":np.array([4.5,1.5]),
    "mu1":np.array([4.2,5]),
    "sig0":0.8,
    "sig1":0.25,
    # potential
    "external": "((cos(y/2)**2)+2*cos(1+x/2)**2)/8",
    # output
    "filename":"particle_data"
}
params_m["N"]=params_m["N1"]+params_m["N0"]
all={}
all["params_m"]=params_m.copy()

# grid for plotting
Ngrid=100
x=np.linspace(0,params_m["Lx"],Ngrid)
y=np.linspace(0,params_m["Ly"],Ngrid)
dv=(params_m["Lx"]*params_m["Ly"])/Ngrid**2
xv, yv = np.meshgrid(x, y, indexing='ij')

# set figure size 
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(9,4),dpi=100)

# collect assort parameters
height=0.1
params_n={"dt":0.00625,
  "sample_rate":20,
  "xv":xv,"yv":yv,
  # contouring levels
  "levels0":np.linspace(0,height,11),
  "levels1":np.linspace(0,height,11),
  "ax":ax,
  }

# initialise particles
# and states
q,p,state=particle_model.init_particles(params_m)
rho0,rho1=particle_model.construct_rho(state, q, params_m, params_n)
# 
sum0=np.sum(rho0)*dv; sum1=np.sum(rho1)*dv
print("Mass of particles\ntype-0: ",sum0,"\ntype-1: ",sum1, "\ntotal: ",sum1+sum0)
#
params_m["mass"]=1/(sum1+sum0)

# time step
nosteps=int(params_m["final_time"]/params_n["dt"])

# counter for time-stepping
count=0
no_frame=0

# storage of output data
all["state"]={}
all["q"]={}
all["time"]={}

# store initial condition
all["state"][0]=state.copy()
all["q"][0]=q.copy()
all["time"][0]=0.0

#
while (count<nosteps):
  count=count+1
  q,p=particle_model.onestep(q,p,params_n["dt"],params_m)
  state=particle_model.reactstep(q,state,params_n["dt"]/2,params_m)
  state=particle_model.reactstep(q,state,params_n["dt"]/2,params_m)
  if (count % params_n["sample_rate"] ==0):
    no_frame=no_frame+1
    all["state"][no_frame]=state.copy()
    all["q"][no_frame]=q.copy()
    all["time"][no_frame]=count*params_n["dt"]
#################
mass=np.zeros(no_frame)
times=np.zeros(no_frame)
# copy mass and times to output data
for k in range(no_frame):
    mass[k]=np.sum(all["state"][k])/params_m["N"]
    times[k]=all["time"][k]
all["no_frames"]=no_frame
all["times"]=times
all["masses"]=mass

# save data to pickle file
with open(params_m["filename"]+'.pkl', 'wb') as f:
    print("Saving to ",params_m["filename"]," pickle")
    pickle.dump(all, f)
    
# animations

def init(): 
    return fig,
#   
def animate(j): 
    ax[0].cla()
    ax[1].cla()
    #print("frame",j)
    t=all["time"][j]
    params_n["title"]=r'$t={:.1f}$'.format(t)
    particle_model.do_plot(all["state"][j],all["q"][j],params_m,params_n)
    return fig,
#
if "gif_animate" in params_n:
    anim = FuncAnimation(fig, animate,  frames = no_frame, interval = 1000, repeat_delay=5000,  blit = True, repeat=False)
    print("\nSaving animation to ", params_m["filename"])
    anim.save(params_m["filename"]+'156.gif',   writer = writer)

# see also output_to_pdf, output_to_anim, output_masses 
# to process outcome

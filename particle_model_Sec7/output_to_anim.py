# -*- coding: utf-8 -*-
"""
Create an animation from data generated by run.py
"""
import importlib
import pickle
#
import numpy as np
#
import scipy
#
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation 


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
writer = PillowWriter(fps=20)#
#
import particle_model

filename="particle_data"
with open(filename+'.pkl', 'rb') as f:
    all = pickle.load(f)
params_m=all["params_m"]

params_m["N"]=params_m["N1"]+params_m["N0"]
# grid for plotting
Ngrid=100
#
x=np.linspace(0,params_m["Lx"],Ngrid)
y=np.linspace(0,params_m["Ly"],Ngrid)
#
dv=(params_m["Lx"]*params_m["Ly"])/Ngrid**2
#
xv, yv = np.meshgrid(x, y, indexing='ij')
#
height=0.14
# collect assort parameters
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(9,4),dpi=100)
#
params_n={"dt":0.00625,
  "sample_rate":20,
  "xv":xv,"yv":yv,
  # contouring levels
  "levels0":np.linspace(0,height,100),
  "levels1":np.linspace(0,height,100),
  "ax":ax,
  }
no_frames=all["no_frames"]
#



def init(): 
    return fig,
   
def animate(j): 
    ax[0].cla()
    ax[1].cla()
    print("frame",j)
    t=all["time"][j]
    params_n["title"]=r'$t={:.1f}$'.format(t)
    particle_model.do_plot(all["state"][j],all["q"][j],params_m,params_n)
    return fig,
   
anim = FuncAnimation(fig, animate,  frames = no_frames, interval = 1000, repeat_delay=5000,  blit = True, repeat=False)
  
print("\nSaving animation to ", params_m["filename"])
anim.save(params_m["filename"]+'1.gif',   writer = writer)

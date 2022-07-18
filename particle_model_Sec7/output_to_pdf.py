# -*- coding: utf-8 -*-
"""
Process data for Section 7
Ourput time snapshots
"""
#import importlib
import pickle
#
import numpy as np
#
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
#
import particle_model
#
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
height=0.1
# collect assort parameters
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(9,4),dpi=100)
#
params_n={"dt":0.00625,
  "sample_rate":20,
  "xv":xv,"yv":yv,
  # contouring levels
  "levels0":np.linspace(0,height,11),
  "levels1":np.linspace(0,height,11),
  "ax":ax,
  }
no_frames=all["no_frames"]
# Choose time shot
j=200
t=all["time"][j]
print("Saving snapshot for t=",t)
#
# set figure size 
ax[0].cla()
ax[1].cla()
print("frame",j)
t=all["time"][j]
params_n["title"]=r'$t={:.1f}$'.format(t)
contours=particle_model.do_plot(all["state"][j],all["q"][j],params_m,params_n)
fig.set_figwidth(5)
fig.set_figheight(3)
fig.colorbar(contours, ax=params_n["ax"][:], location='right', shrink=0.5)
fig.savefig(filename+str(j)+'.pdf', bbox_inches="tight")
# Firedrake RIDK 2d interact
# ====================================
#
# RIDK equation (without a potential) in 2d
#  on a periodic domain
#
# rho_t=-div j
# j_t = -gamma j - c grad rho -rho (grad V_pair * rho) - grad V_ext rho+ (sigma/root(N)) sqrt(rho) xi
#
# for white-in-time and spatially correlated noise xi with length scale epsilon and c=sigma^2/2 gamma
# with parameters (gamma,sigma,N,epsilon)
# V_pair is a radial pair potential;
#  V_ext is an external potentl
#
import importlib
import pickle
import sys
#from tkinter.tix import Tree
if len(sys.argv)>1:
  run_case=int(sys.argv[1])
  print("run_case", run_case)
else:
  run_case=0
  print("Default params")
#
import matplotlib as mpl
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
#
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation 
writer = PillowWriter(fps=10)
#
#
import RIDK
from RIDK import *
importlib.reload(RIDK)
#
linear_solver={}

######### base dicts
params_m = {  # model parameters
    "gamma": 0.3,
    "N1": 4.5e3,
    "N2": 5e2,
    "mass":1,
    "sigma": 0.2,
    "Lx": 2*pi,
    "Ly": 2*pi,
    "epsilon": 0.2,
    "MD_rate":0.2,
    "reaction_radius":0.15,
    "initial_rho": berlin_rho1_2d,
    "initial_j": zero_j_2d,
    "final_time": 25,#25
    "filename": "RIDK-2d-interact",
    "external": "((cos(y/2)**2)+2*cos(1+x/2)**2)/8",
}
params_n = {  # numerical-discretisation parameters
    ################################################
    "degree": 0,  # of elements
    "element_noise": "DG",
    "element_rho": "DG",
    "element_j": "Raviart-Thomas",
    "weak_form": weak_form_em,
    "noise_degree": 0,  # of elements
    "theta": 0.5,
    "solver_parameters": linear_solver,
    "no_x": 50,#100
    "no_y": 50,#100
    "no_t": 4000,#4000
    "convolution_radius": pi / 2,
    "noise_truncation": 80,
    "screen_output_steps": 20,    
    "SIPG_D":0., # extra diffusion term
    "rho_thresh":0, 
    "tau":0, # time-scale regularisation
}
if run_case==2:
  tau_in=0.05
  filename="RIDK-2d-interact-tau"
  print("tau:", tau_in," (time-scale reg) filename:", filename)
  params_n["tau"]=tau_in
  params_m["filename"]=filename
####
params_m["N"]=params_m["N1"]+params_m["N2"]
params_m1=params_m.copy()
params_m2=params_m.copy()
all={}
tmp=params_m.copy()
tmp["initial_rho"]="0"
tmp["initial_j"]="0"
tmp["weak_form"]="0"
all["params_m"]=tmp
# 
params_m1["mass"]=params_m["N1"]/params_m["N"]
params_m2["mass"]=params_m["N2"]/params_m["N"]
assert abs(params_m1["mass"]+params_m2["mass"]-1)<1e-14

######### field 1
params_m1["source"] =True # 1 is converted to 2
params_n1=params_n.copy()
# Set-up mesh etc
get_test_trial_space(params_m1, params_n1)
# Set-up initial data and exact soln
rho01, j01 = set_up_2d(params_m1, params_n1)
tmp=rho01.copy(True)
########### field 2
params_m2["source"] =False
params_n2=params_n.copy()
params_m2["initial_rho"] =berlin_rho2_2d
# Set-up mesh etc
get_test_trial_space(params_m2, params_n2)
# Set-up initial data and exact soln
rho02, j02 = set_up_2d(params_m2, params_n2)
##################################################
# define  weak form and time-stepper
# all the work is done here
one_step1, dt = params_n1["weak_form"](params_m1, params_n1, rho01, j01)
one_step2, dt = params_n2["weak_form"](params_m2, params_n2, rho02, j02)
#
t = 0.0
step = 0
# masses
total_mass01 = assemble(rho01 * dx)
total_mass02 = assemble(rho02 * dx)
print("Inital mass",total_mass01+total_mass02)
# for plotting
V1 = FunctionSpace(params_n1["mesh"], "CG", 1)
plot1 = Function(V1)
V2 = FunctionSpace(params_n2["mesh"], "CG", 1)
plot2 = Function(V2)
data1={}
data2={}
# store initial conditio
plot1.interpolate(rho01)
data1[0]=plot1.copy(True)
plot2.interpolate(rho02)
data2[0]=plot2.copy(True)
i=0
height0=0.1
height1=0.1
no_levels0=11
no_levels1=no_levels0#int(no_levels0*height1/height0)
levels0 = np.linspace(0, height0, no_levels0)
levels1 = np.linspace(0, height1, no_levels1)
levelsneg= np.linspace(-height1, -1e-3, no_levels1)
#
cmap1 = mpl.cm.get_cmap("Greens").copy()
cmap2 = mpl.cm.get_cmap("Blues").copy()
cmap1=cmap2
###################################################
#
all["data2"]={}
all["data1"]={}
all["time"]={}
all["mass1"]={}
all["mass1"][0]=assemble(rho02 * dx)
#
while t < params_m1["final_time"] - dt:
    tmp.assign(rho01)#.copy(True))
    # update rho01
    one_step1(rho02)
    #update rho02
    one_step2(tmp)
    step += 1
    t += dt
    # regularly print to screen and save to file
    if step % params_n1["screen_output_steps"] == 0:
        i=i+1
#        outfile.write(rho01)
        total_mass = assemble(rho01 * dx)+assemble(rho02 * dx)
        mass_deviation = total_mass - total_mass01-total_mass02
        plot1.interpolate(rho01)
        data1[i]=plot1.copy(True)
        plot2.interpolate(rho02)
        data2[i]=plot2.copy(True)
        all["mass1"][i]=assemble(rho02 * dx)
        print(
            "t={:.3f}".format(t),
            "; total_mass={:.3f}".format(total_mass),
            ", deviation={:.2e}".format(mass_deviation),
             ", min={:.2e}".format(rho01.dat.data.min()),
            ", {:.2e}".format(rho02.dat.data.min()),
          ", max={:.1e}".format(rho01.dat.data.max()),
            ", {:.1e}          ".format(rho02.dat.data.max()),
            end=" \r",
        )
#        print('f min: {:.3g}, max: {:.3g} '.format(f.dat.data.min(), f.dat.data.max()))
        if mass_deviation > 1:
            print("\n")
print("frames",i)
all["no_frames"]=i
all["times"]=np.zeros(i)
all["masses"]=np.zeros(i)
for k in range(i):
  all["masses"][k]=all["mass1"][k]
  all["times"][k]=k*params_n["screen_output_steps"]*dt
#  
with open(params_m["filename"]+'.pkl', 'wb') as f:
    print("Saving to ",params_m["filename"]," pickle")
    pickle.dump(all, f)
    

print("\n")
pdf_output=True
gif_output=True
my_frames=[0,10,20,30,40,50,60,70,80,90,i]
#my_frames=[0]
if (pdf_output==True):
  #print("Save frame",i)
  for j in my_frames:
    t=j*params_n["screen_output_steps"]*dt
    params_n["title"]=r'$t={:.1f}$'.format(t)
    fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(5,3),dpi=100)
    #
    firedrake.tricontourf(data1[j], axes=axes[0],levels=levels0,  cmap=cmap1, extend='max')
    contours=firedrake.tricontourf(data2[j], axes=axes[1],levels=levels1,  cmap=cmap2, extend='max')

    firedrake.tricontourf(data1[j], axes=axes[0],levels=levelsneg,  cmap='Reds_r', extend='min')
    firedrake.tricontourf(data2[j], axes=axes[1],levels=levelsneg,  cmap='Reds_r', extend='min')
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    #axes[0].set_title("PDE")
    #axes[1].set_title(params_n["title"])
    axes[0].set_xlabel(r'$x$')
    axes[1].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')
    
    fig.colorbar(contours, ax=axes[:], location='right', shrink=0.5)
    #plt.ylim(0,height)
    fig.savefig(params_m["filename"]+str(j)+'.pdf', bbox_inches="tight")

############

def init(): 
  
    return fig,
   
def animate(j):
    t=j*params_n["screen_output_steps"]*dt
    params_n["title"]=r'$t={:.1f}$'.format(t)
    axes[0].cla()
    axes[1].cla()
    # reaction 2+1->1
    firedrake.tricontourf(data1[j], axes=axes[0],levels=levels0,  cmap=cmap1, extend='max')
    firedrake.tricontourf(data2[j], axes=axes[1],levels=levels1,  cmap=cmap2, extend='max')

    firedrake.tricontourf(data1[j], axes=axes[0],levels=levelsneg,  cmap='Reds_r', extend='min')
    firedrake.tricontourf(data2[j], axes=axes[1],levels=levelsneg,  cmap='Reds_r', extend='min')
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    # axes[0].axis('off')
    # axes[1].axis('off')
    
    axes[0].set_title("PDE")
    axes[1].set_title(params_n["title"])
    axes[0].set_xlabel(r'$x$')
    axes[1].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')
    axes[1].set_ylabel(r'$y$')
    return fig,
   
if (gif_output==True):
  anim = FuncAnimation(fig, animate,  frames = i, interval = 1000, blit = True, repeat=False)
  print("Saving animation to ", params_m["filename"])
  anim.save(params_m["filename"]+'.gif',   writer = writer)
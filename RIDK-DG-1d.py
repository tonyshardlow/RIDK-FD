# Firedrake RIDK 1d
# ====================================
#
# RIDK equation (without a potential) in 1d
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
import os
import sys
############
if len(sys.argv)>1:
  run_case=int(sys.argv[1])
  print("run_case", run_case)
else:
  run_case=0
  print("Default params")
##
import RIDK
from RIDK import *
importlib.reload(RIDK)
############
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation 
import tikzplotlib
writer = PillowWriter(fps=30)

##################################
# parameters
#
linear_solver={}
# linear_solver= {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"           }
# # {"ksp_type": "preonly", "pc_type": "lu"}
#
params_m = {  # model parameters
    "gamma": 0.25,
    "N": 1e3,
    "sigma": 0.25,
    "mass":1,
    "reaction_rate":0.0,
    "Lx": 2.0 * pi,
    "epsilon": 0.05,
    "initial_rho":  example_rho_1d,
    "initial_j": zero_j,
    "final_time": 10,
    "reaction_rate":0.0,
    # "pair": '-cos(pi*r)',
    "external": "0.5*cos(x)**2",
    "filename":"RIDK_1d_default"
}
params_n = {  # numerical-discretisation parameters
    ################################################
    "degree": 1,  # of elements
    "noise_degree": 1,  # of elements
    "element_noise": "DG",
    "element_rho": "CG",
    "element_j": "CG",
    "SIPG_D":0., # extra diffusion term
    "SIPG_reg": 1,
    "weak_form": weak_form_em,
    "delta":0,
    "solver_parameters": linear_solver,
    "no_x": 400,
    "no_t": 800,
    "convolution_radius": pi / 2,
    "noise_truncation": 80,
    "screen_output_steps": 10,
    "rho_thresh":0, 
    "tau":0, # phi regularisation
}
if run_case==2:
  tau_in=0.2
  filename="fig_tau"
  print("tau:", tau_in," (time-scale reg) filename:", filename)
  params_n["tau"]=tau_in
  params_m["filename"]=filename
if run_case==21:
  tau_in=0.05
  filename="fig_tau_small"
  print("tau:", tau_in," (time-scale reg) filename:", filename)
  params_n["tau"]=tau_in
  params_m["filename"]=filename
if run_case==1: # Example 6.1
  D_in=0.5
  filename="fig_diffusion"
  print("diffusion:", D_in," (diffusion reg) filename:", filename)
  params_n["SIPG_D"]=D_in
  params_m["filename"]=filename
if run_case==11: # Example 6.1
  D_in=0.1
  filename="fig_diffusion_small"
  print("diffusion:", D_in," (diffusion reg) filename:", filename)
  params_n["SIPG_D"]=D_in
  params_m["filename"]=filename


# Set-up mesh etc
get_test_trial_space(params_m, params_n)
# Set-up initial data and exact soln
rho0, j0 = set_up_1d(params_m, params_n)
# define  weak form and time-stepper
# all the work is done here
one_step, dt = params_n["weak_form"](params_m, params_n, rho0, j0)
print("done")
#
V = FunctionSpace(params_n["mesh"], "CG", 1)
plot1 = Function(V)
#
t = 0.0
step = 0
#
data1={}
tikz_data={}
tikz_data[0]=rho0.copy(True)
i=0
#
total_mass0 = assemble(rho0 * dx)
#
while t < params_m["final_time"] - dt:
    #print("here",rho0.at(0.5))
    tmpt = one_step(rho0)
    step += 1
    t += dt
    # regularly print to screen and save to file
    if step % params_n["screen_output_steps"] == 0:
        #outfile.write(rho0)
        i=i+1
        total_mass = assemble(rho0 * dx)
        mass_deviation = total_mass - total_mass0
        plot1.interpolate(rho0)
        tikz_data[i]=rho0.copy(True)
        data1[i]=plot1.copy(True)
        print(
            "t={:.3f}".format(t),
            "; total_mass={:.3f}".format(total_mass),
            ", deviation={:.3e}".format(mass_deviation),
            end=" \r",
        )
        if mass_deviation > 1:
            print("\n")
print("\n")
fig, axes = plt.subplots()
#
if params_n["element_rho"] == "DG":
    print("Interpolating DG rho...", end="")
    V = FunctionSpace(params_n["mesh"], "CG", 1)
    u = Function(V)
    u0 = Function(V)
    if not (tmpt is None):
        u.interpolate(tmpt[0] / 5 + 1)
        plot(u, axes=axes, label="rho_con", color="b")
    u0.interpolate(rho0)
    plot(u0, axes=axes, label="numerical", color="r")
    print("..finished")
else:
    for j in range(0,i+1,int((i+1)/4)):
      print(j, " of ", i)
      plot(tikz_data[j], axes=axes,  color="b")
#
axes.legend()
axes.set_xlabel(r'$x$')
axes.set_ylabel(r'$\rho$')
axes.grid(True, which='both')
axes.set_ylim(-0.2,1)
fig.show()
fig.set_figwidth(6)
fig.set_figheight(4)
fig.savefig("fig1.pdf", bbox_inches="tight")
tikzplotlib.save("test.tex",axis_width=r'\awidth',axis_height=r'\aheight')#,standalone=True)
with open("tmp.tex", 'w') as f: 
    for key, value in params_n.items(): 
        f.write('%% %s:%s\n' % (key, value))
    for key, value in params_m.items(): 
        f.write('%% %s:%s\n' % (key, value))
os.system(r'cat  start.tex test.tex tmp.tex close.tex >'+ params_m["filename"]+'.tex')
os.system('cp '+params_m["filename"]+'.tex /mnt/c/Users/tony/Documents/GitHub/RIDK-DG/figs')

#os.system(r'cat  tmp.tex  close.tex> out.tex')
#

def init(): 
  
    return fig,
   
def animate(j):
    x = np.linspace(0, 1, 1000)
    axes.cla()
    plot(data1[j+1],axes=axes,color="b")
    plt.ylim(-0.2,1)
    #plt.set_aspect('equal')
    plt.grid(True, which='both')
    axes.set_xlabel(r'$x$')
    axes.set_ylabel(r'$\rho$')
      
    return fig,
   
anim = FuncAnimation(fig, animate,  frames = i, interval = 1000, blit = True, repeat=False)
  
filename=params_m["filename"]+'.gif'
print("\nSaving animation to ", filename)
anim.save(filename,   writer = writer)
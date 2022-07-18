# Firedrake RIDK 2d
# ====================================
#
# RIDK equation
# on a periodic domain in 2d
#
# rho_t=-div j
# j_t = -gamma j - c grad rho -rho (grad V_pair * rho) - grad V_ext rho+ (sigma/root(N)) sqrt(rho) xi
#
# for white-in-time and spatially correlated noise xi with length scale epsilon and c=sigma^2/2 gamma,
# with parameters (gamma,sigma,N,epsilon)
# V_pair is a radial pair potential
# V_ext is an external potentl
#
import importlib
import os
import sys
#
if len(sys.argv)>1:
  run_case=int(sys.argv[1])
  print("run_case", run_case)
else:
  run_case=0
  print("Default params")
#############################
import matplotlib as mpl
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'lmodern'
mpl.rcParams['font.sans-serif'] = 'cm'
import matplotlib.pyplot as plt
import tikzplotlib
###########################
#
#export OMP_NUM_THREADS=1
import firedrake
#
import RIDK
from RIDK import *
importlib.reload(RIDK)

#
linear_solver={}
#linear_solver = {"ksp_type": "preonly", "pc_type": "lu"}
#linear_solver         = {"ksp_type": "preonly", "pc_type": "bjacobi", "sub_pc_type": "ilu"}
#
params_m = {  # model parameters
    "gamma": 0.25,
    "N": 3e3,
    "mass":1,
    "sigma": 0.4,
    "reaction_rate":0.0,
    "Lx": 2.0 * pi,
    "Ly": 2.0 * pi,
    "epsilon": 0.05,
    "final_time": 1,
    "initial_rho": uniform_rho_2d,
    "initial_j": zero_j_2d,
   # "pair": "r**2",
    #"c_to_one": True,
    "external": "0.5*(cos(x)**2+cos(y)**2)",
    "filename":"fig_2d"
}
#
params_n = {  # numerical-discretisation parameters
    "degree": 0,  # of elements
    "element_rho": "DG",
    "element_j": "Raviart-Thomas",
    "noise_degree": 0,  # of elements
    "element_noise": "CG",
    "SIPG_D":0., # extra diffusion term
    "SIPG_reg": 1,
    "weak_form": weak_form_em,
    "solver_parameters": linear_solver,
    "no_x": 150,
    "no_y": 150,
    "no_t": 101,
    "screen_output_steps": 20,
    "convolution_radius": 0.1,
    "noise_truncation": 40,
    "rho_thresh":0, 
    "tau":0, # phi regularisation
}
kbT=params_m["sigma"]**2/(2*params_m["gamma"]); 
gdx=params_m["Lx"]/params_n["no_x"]; gdt=params_m["final_time"]/params_n["no_t"]; wave_speed=sqrt(kbT)
cfl=wave_speed*gdt/gdx
print("CFL constant:", cfl,"\n\n")
if (cfl>1):
  exit()

if run_case==1:
  tau_in=0.025
  filename="fig_tau_small"
  print("tau:", tau_in," (time-scale reg) filename:", filename)
  params_n["tau"]=tau_in
  params_m["filename"]=filename
# Set-up mesh etc
get_test_trial_space(params_m, params_n)
# Set-up initial data
rho0, j0 = set_up_2d(params_m, params_n)
# define weak form and time-stepper
# all the work is done here
one_step, dt = params_n["weak_form"](params_m, params_n, rho0, j0)
#
t = 0.0
step = 0
coeffs = [0.0, 0.0]
#
height=0.1
levels = np.linspace(0, height, 11)
cmap = "hsv"
total_mass0 = assemble(rho0 * dx)
print("Initial mass", total_mass0)
#
print("Starting time-stepping..")
rt = time.perf_counter()
fig, axes = plt.subplots(nrows=1, ncols=2)
axes_flat=axes.flat
#
while t < params_m["final_time"] - dt:
    one_step(rho0)
    #
    step += 1
    t += dt
    # regularly print to screen and save to file
    if step % params_n["screen_output_steps"] == 0:
     #   outfile.write(rho0)
        total_mass = assemble(rho0 * dx)
        mass_deviation = total_mass - total_mass0
        print(
            "t={:.3f}".format(t),
            "; run_time {:.1f}".format(time.perf_counter() - rt), 
            "secs; total_mass={:.3f}".format(total_mass),
            ", deviation={:.3e}".format(mass_deviation),
            end=" \r")
        if abs(mass_deviation )> 1:
            print("\n")
            exit()
    if step % 50 == 0:
        axnum=(step//50)-1
        print("axnum",axnum)
     #   outfile.write(rho0)
        print("save:",t,"\n")

        print(rho0.dat)
        contours = firedrake.tricontourf(rho0, levels=levels, axes=axes[axnum], cmap=cmap)
        axes[axnum].set_aspect("equal")
#        axes.set_xlabel(r'$x$')
#        axes.set_ylabel(r'$y$')
        axes[axnum].axis('off')
#
fig.colorbar(contours, ax=axes[:], location='right', shrink=0.5)
 
#
os.system(r'cat  start2.tex test.tex tmp.tex close.tex >'+ params_m["filename"]+'.tex')
os.system('cp '+params_m["filename"]+'.tex /mnt/c/Users/tony/Documents/GitHub/RIDK-DG/figs')
fig.savefig("fig1.pdf", bbox_inches="tight",dpi=50)
os.system('cp fig1.pdf /mnt/c/Users/tony/Documents/GitHub/RIDK-DG/figs/'+params_m["filename"]+'_.pdf')
print("\n")
fig.show()
#
wait = input("Press Enter to continue.")

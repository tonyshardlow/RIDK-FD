# RIDK 1d 
# reacting/diffusing 
#
import importlib
#
import RIDK
from RIDK import *
importlib.reload(RIDK)
#
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation 
writer = PillowWriter(fps=30)
#
linear_solver={}

# RIDK parameter data
params_m = {  # model parameters
    "gamma": 1,
    "N": 3e3,
    "sigma": 0.25,
    "mass":1,
    "Lx": 1.,
    "epsilon": 0.01,
    "reaction_rate":2.0,
    "initial_rho": berlin_rho2,
    "initial_j": zero_j,
    "final_time": 4,
    "external": "0.05*cos(x*2*pi)**2",
}
params_n = {  # numerical-discretisation parameters
    ################################################
    "degree": 1,  # of elements
    "noise_degree": 0,  # of elements
    "element_noise": "DG",
    "element_rho": "DG",
    "element_j": "CG",
    "SIPG_D":0., # extra diffusion term
    "SIPG_reg": 1,
    "weak_form": weak_form_em,
    "theta": 0.5,
    "solver_parameters": linear_solver,
    "no_x": 100,
    "no_t": 2000,
    "convolution_radius": pi / 2,
    "noise_truncation": 80,
    "screen_output_steps": 10,    
    "rho_thresh":0, 
    "tau":0, # phi regularisation
}
######### field 1
params_m1=params_m.copy()
params_m1["source"] =True
params_n1=params_n.copy()
# Set-up mesh etc
get_test_trial_space(params_m1, params_n1)
# Set-up initial data and exact soln
rho01, j01 = set_up_1d(params_m1, params_n1)
tmp=rho01.copy(True)
########### field 2
params_m2=params_m.copy()
params_m2["source"] =False
params_n2=params_n.copy()
params_m2["initial_rho"] =berlin_rho1
# Set-up mesh etc
get_test_trial_space(params_m2, params_n2)
# Set-up initial data and exact soln
rho02, j02 = set_up_1d(params_m2, params_n2)
# define  weak form and time-stepper
# all the work is done here
one_step1, dt = params_n1["weak_form"](params_m1, params_n1, rho01, j01)
one_step2, dt = params_n2["weak_form"](params_m2, params_n2, rho02, j02)
#
t = 0.0
step = 0
#
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
i=0
#
while t < params_m1["final_time"] - dt:
    tmp.assign(rho01)#.copy(True))
    # update rho01
    tmpt = one_step1(rho02)
    #update rho02
    tmpt = one_step2(tmp)
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
        print(
            "t={:.3f}".format(t),
            "; total_mass={:.3f}".format(total_mass),
            ", deviation={:.3e}".format(mass_deviation),
            end=" \r",
        )
        if mass_deviation > 1:
            print("\n")
print("\n")

######################################
## plotting
##
fig, axes = plt.subplots()
#
plot(data2[i],axes=axes)
fig.show()
#
axes.legend()
plt.ylim(0,5)
fig.show()
fig.savefig("fig1.pdf", bbox_inches="tight")

# for animations
def init():  
    return fig,
   
def animate(j):
    x = np.linspace(0, 1, 1000)
    axes.cla()
    plot(data2[j+1],axes=axes,color="r")
    plot(data1[j+1],axes=axes,color="b")
    plt.ylim(0,5)
    #fig.show()
    #plot(data2[i])
    # plots a sine graph

      
    return fig,
   
anim = FuncAnimation(fig, animate,  frames = i, interval = 1000, blit = True, repeat=False)
  
filename='RIDK-interact.gif'
print("Saving animation to ", filename)
anim.save(filename,   writer = writer)
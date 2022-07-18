# RIDK.py
# 
# library of routines for DG approximation
# of RIDK stochastic PDE using Firedrake
#
# Load libraries
#
import time
import math
#
from firedrake import *
#
firedrake.logging.set_level(100)
#
import numpy as np
#
import scipy
from scipy import sparse
from scipy.spatial import cKDTree
from scipy import stats
#
import sympy
from sympy.utilities.lambdify import lambdify
#
import matplotlib.pyplot as plt
# FFT
import finufft

###################################################
def evalue(n: int, model):
    # get the nth eigenvalue of linear part (choose one of the pair)
    # 1d only
    assert model["c_to_one"], "Set parameter 'c_to_one' to 'True'"  # only works for c=1
    gamma = model["gamma"]
    d = gamma ** 2 - 4 * n ** 2
    if d >= 0:
        lam = -(gamma + np.sqrt(np.real(d))) / 2  # case = "real"
    else:
        lam = -(gamma + np.sqrt(np.complex(d))) / 2  # case = "complex"
    return lam


###################################################
def efun(n, y, model):
    # get an example eigenfunction corresponding to nth eigenvalue
    # y is a parameter controlling choice of eigenfunction
    # return eigen-value and -function for rho, j components
    # and, for rho, return real and imaginary parts separately.
    #
    # assert y in [0,1] for convex combination of eigenvectors
    # 1d only
    assert y >= 0 and y <= 1, "Invalid y"
    # grab eigenvalue
    lam = evalue(n, model)
    # two families according to type of lambda
    if type(lam) == real:  # real eigenvalue
        c = [-n / lam, n / (lam + gamma)]
        # rho eigenfunction
        def fr(x):
            return c[0] * y * cos(n * x) + c[1] * (1 - y) * cos(n * x)

        def fi(x):
            return 0  # trivial imaginary part

        # j eigenfunction
        def g(x):
            return sin(n * x)

    else:  # complex eigenvalue
        c = -n / lam
        # rho eigenfunction
        def fr(x):
            return np.real(c) * (y * cos(n * x) - (1 - y) * sin(n * x))

        def fi(x):
            return np.imag(c) * (y * cos(n * x) - (1 - y) * sin(n * x))

        # j eigenfunction
        def g(x):
            return y * sin(n * x) + (1 - y) * cos(n * x)

    print("Eigenvalue lambda=", lam)
    return lam, fr, fi, g


#######################################################
def exact_rho(lam, mass, eig_fn_rho, eig_fn_rho1):
    # return an exact solution in rho (matching exact_j)
    # 1d only
    if type(lam) == real:
        return lambda t, x: exp(lam * t) * eig_fn_rho(x) + mass
    else:
        lamr = np.real(lam)
        lami = np.imag(lam)
        return (
            lambda t, x: exp(lamr * t)
            * (cos(lami * t) * eig_fn_rho(x) - sin(lami * t) * eig_fn_rho1(x))
            + mass
        )


#######################################################
def exact_j(lam, wind, eig_fn_j):
    # return an  exact solution in j (matching exact_rho)
    # 1d only
    if type(lam) == real:
        return lambda t, x: exp(lam * t) * eig_fn_j(x) + wind
    else:
        lamr = np.real(lam)
        lami = np.imag(lam)
        return lambda t, x: exp(lamr * t) * (cos(lami * t) * eig_fn_j(x)) + wind


#############################
def Print_L2_error(params_m, t, rho, j, params_n):
    # Assuming exact solution is set-up, print L2 error to screen
    # 1d only
    V_rho = params_n["trial_rho"]
    V_j = params_n["trial_j"]
    mesh = params_n["mesh"]
    (x,) = SpatialCoordinate(mesh)
    # print L2 errors to screen 
    def tmp(x):
        return params_m["exact_rho"](t, x)

    exact_rho = Function(V_rho).interpolate(tmp(x))

    def tmp(x):
        return [params_m["exact_j"](t, x)]

    exact_j = Function(V_j).interpolate(as_vector(tmp(x)))
    #
    L2_err_rho = sqrt(assemble((rho - exact_rho) * (rho - exact_rho) * dx))
    L2_err_j = sqrt(assemble(dot(j - exact_j, j - exact_j) * dx))
    L2_norm_rho = sqrt(assemble(exact_rho * exact_rho * dx))
    L2_norm_j = sqrt(assemble(dot(exact_j, exact_j) * dx))
    #
    print(
        "t=%6.3f," % (t),
        " rel. L2-error (rho,j)=(%8.2E," % (L2_err_rho / L2_norm_rho),
        "%8.2E)," % (L2_err_j / L2_norm_j),
        " abs. L2-error=%8.2E" % (L2_err_rho + L2_err_j), end="  \r"
    )
    return exact_rho


#
def example_rho(x, y):
    # example initial data for rho in 2d
    return 2 + 0.4*( cos( x+ y))

#
def uniform_rho_2d(x, y):
    # example initial data for rho in 2d
    return Constant(1/(4*pi*pi))
#
def example_j(x, y):
    # example initial data for j in 2d
    tmp = example_rho(x, y)
    return (0.5 * tmp, -0.2 * tmp)
#
def zero_j_2d(x, y):
    # example initial data for j in 2d
    return [Constant(0.0),Constant(0.0)]
#
def uniform_rho(x):
    # initial data  for rho (uniform distribution)
    c=4*pi*pi
    return (1/c)
#
def example_rho_1d(x):
    # example initial data for rho in 1d
    c=2*pi*(1+pi)
    return (1 + x)/c
#
def uniform_rho2(x):
    # initial data  for rho (uniform distribution)
    return (1.0)
#
def berlin_rho1(x):
  mu=0.5; sig_sq=0.2**2
  return exp(-(x-mu)**2/(2*sig_sq))
#
def berlin_rho2(x):
  mu=0.7; sig_sq=0.05**2
  return  exp(-(x-mu)**2/(2*sig_sq))
#
def berlin_rho1_2d(x,y):
  mu_x=4.5; mu_y=1.5; sig_sq=0.8**2;
  return conditional((x-mu_x)**2+(y-mu_y)**2<2*sig_sq, exp(-(x-mu_x)**2/(2*sig_sq))*exp(-(y-mu_y)**2/(2*sig_sq)),0)
#
def berlin_rho2_2d(x,y):
  mu_x=4.2; mu_y=5; sig_sq=0.25**2; 
  return  conditional((x-mu_x)**2+(y-mu_y)**2<2*sig_sq, exp(-(x-mu_x)**2/(2*sig_sq))*exp(-(y-mu_y)**2/(2*sig_sq)),0)
#
def zero_j(x):
    # initial data for j (zero momentum)
    return [Constant(0.0)]
#
def get_bessel_ratios(N, ep, print_errors=False):
  # return vector of I_j(1/2ep^2)/I_0(1/2ep^2) for j=0,...,N-1
  # for coefficients of noise
  # subtle to rounding errors in the ratio
  # Use symbolic
  x=sympy.Symbol('x')
  epsilon=sympy.Symbol('epsilon')
  # Large x expression for I_1(x)/I_0(x)
  f=(1-3/(8*x)*(1+5/(16*x)))/(1+1/(8*x)*(1+9/(16*x)))
  # we want to evaluate at x=1/(2 ep**2)
  fep=f.subs(x,(0.5/epsilon**2))
  # convert to a callable function
  fepc = sympy.lambdify(epsilon, fep)    
  # for testing purposes
  def rrr(x,j): return scipy.special.ive(j,x)/scipy.special.ive(0,x)
  #
  def ratio_bessel(ep):
    # evalute I_1(1/2ep*2)/I_0(1/2ep**2) for any eps>0 accuarately
    if ep<1e-3: # switch for asymptotic expansions
      return fepc(ep)
    else:
      return rrr(0.5/ep**2,1)
    #
  X=np.empty(N)
  X[0]=1
  X[1]=ratio_bessel(ep)
  #
  switch=False
  for i in range(1,N-1):
    # recurrence relation        
    # r_{j+1}=r_{j-1}-4*j*ep**2*r_j  
    if X[i-1]/X[i]-4*i*(ep**2)<10*ep or switch: # recurrence relation breaks down due to rounding
      X[i+1]=rrr(0.5/ep**2,i+1)
      switch=True
    else: # use recurrence relation
      X[i+1]=X[i]*(X[i-1]/X[i]-4*i*(ep**2))
  if print_errors:
    print("expansion for Bessel ratio",fep.simplify())
    for i in range(N):
       print("i=",i, ", X[i]=",X[i],", error=",abs(X[i]- rrr(0.5/ep**2,i)))
  #
  return X
#
#
def get_noise_coeff(ep, L, N):
    # compute coefficients for Fourier transform for noise sample (1d)
    # with N modes (requires N odd) on domain width L and parameter 
    # epsilon=ep
    assert N % 2 == 1  # assert N odd
    # rescale ep
    ep = (ep / L) * (2 * pi)
    # compute ratio of Bessel functions
    ratios = get_bessel_ratios(N, ep)
    # wave numbers
    k = np.arange(N) - int((N - 1) / 2)
    # eigenvalues of covariance
    lamk = ratios[abs(k)]
    # coefficients
    noise_coeff = np.sqrt(lamk * 2 / L)
    noise_coeff[int((N - 1) / 2)] = np.sqrt(1 / L)  # corresponding to k=0
    return noise_coeff
#
#
def get_ext_reaction(params_m, params_n):
  z = Function(params_n["trial_rho"])
  def set_ext_reaction(source_in):
    z.assign(source_in)
    #z.dat.data[:]=source_in.dat.data[:]
#    print("after", zsource.at(0.5))
  return z, set_ext_reaction  
#
def get_noise(params_m, params_n):
    # return a function definition and xi
    # call the function to generate a new sample in xi (of noise)
    if params_m["sigma"] < 1e-16:

        def sample():  # dummy function
            return True

        return None, sample
    #
    W = params_n["noise_space"]
    m = W.ufl_domain()  # mesh
    X = interpolate(m.coordinates, W)
    # Make an output function.
    xi = Function(params_n["noise_space"])
    # define Fourier coefficients
    N = int(2 * np.ceil(params_n["noise_truncation"] / 2.0)) + 1
    ep = params_m["epsilon"]
    if "no_y" in params_n and "Ly" in params_m:  # 2d
      # translate to [-pi,pi]x[-pi,pi]
      X_values = np.einsum(
        "ij,j->ij",
         X.dat.data_ro,
         (2*pi) /
          np.array([params_m["Lx"], params_m["Ly"]])
         )-np.array([[pi,pi]])

      noise_coeff_2d = np.outer(
            get_noise_coeff(ep, params_m["Lx"], N),
            get_noise_coeff(ep, params_m["Ly"], N),
        )

      def sample():
        xi_k = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        a_k = noise_coeff_2d * xi_k
        # fast Fourier transform
        mydata = finufft.nufft2d2(X_values[:, 0], X_values[:, 1], a_k)
        xi.dat.data[:, 0] = np.real(mydata)
        xi.dat.data[:, 1] = np.imag(mydata)

    else:  # 1d
        # translate to [-pi,pi]
        X_values = X.dat.data_ro * (2 * pi) / params_m["Lx"] - pi
        noise_coeff = get_noise_coeff(ep, params_m["Lx"], N)
        def sample():
            xi_k = np.random.randn(N) + 1j * np.random.randn(N)
            a_k = noise_coeff * xi_k
            #  fast Fourier transform
            mydata = finufft.nufft1d2(X_values.flatten(), a_k)
            #  external data function to interpolate the values
            xi.dat.data[:] = np.real(mydata)  # lost imaginary part (TODO) !!!!

    return xi, sample

#############
def convolution_slow(params_n):
    # in progress - for convolution term V*rho
    V = params_n["trial_rho"]
    m = V.ufl_domain()  # mesh
    # Now make the VectorFunctionSpace corresponding to V.
    W = VectorFunctionSpace(m, V.ufl_element())
    # Next, interpolate the coordinates onto the nodes of W.
    X = interpolate(m.coordinates, W)
    X_values = X.dat.data_ro
    n = X_values.shape[0]
    # Make an output function.
    ins = Function(V)
    out = Function(V)
    #
    r = 0.1  # cutoff for loop
    if "no_y" in params_n and "Ly" in params_m:  # 2d
        weight = 2 * pi * r ** 2
    else:
        weight = 2 * r
    #
    t = time.perf_counter()
    print("  Creating convolution matrix...",end="")
    mtx = sparse.dok_matrix((n, n))
    count = np.zeros(n)  # for normalisation
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(X_values[i] - X_values[j])
            if d < r:
                count[i] += 1
                count[j] += 1
                mtx[i, j] = Vdash(d) * weight
                mtx[j, i] = Vdash(d) * weight
    for i in range(n):
        mtx[i, :] /= count[i]  # rescale
    mtx = mtx.tocsr()  # convert to row-based format convenient for matrix multiply
    print("finished in {:.3f}".format(time.perf_counter() - t))

    def do_convolution(in_fn, out_fn):
        out_fn.dat.data[:] = mtx * in_fn.dat.data[:]

    # test mtx
    return do_convolution


#############
def Vdash(d):
    # example function for derivative of radial pair potential
    return np.where(d < 1, d ** 2, 0.0)
    #############


def Vext(Xvalues):
    # example function for derivative of radial pair potential
    return np.zeros(Xvalues.shape)


#############
def str_radial_pair_pot_to_force(params_m):
    # convert radial pair potential as a string in r (e.g., 'r**2')
    # to a function
    r = sympy.symbols("r")
    Vr = params_m["pair"]
    print("Radial pair pot ", Vr)
    Vrd = sympy.diff(Vr, "r")
    # radial
    func_numpy = lambdify(r, Vrd, modules=[np], dummify=False)
    # alternative implementation - speed not important here
    #   func_numexpr = lambdify(r, Vrd, modules=[numexpr], dummify=False)
    #
    def radial_pair_deriv(X):
        return np.array(func_numpy(X))

    return radial_pair_deriv


#############
def str_ext_pot_to_force(params_m):
    # convert external potential as a string in x,y (e.g., 'x**2 + y**2')
    # to a function
    x, y = sympy.symbols("x y")
    Vext = params_m["external"]
    func_numpy = lambdify((x, y), Vext, modules=[np], dummify=False)
    #
    assert (abs(func_numpy(0, 0) - func_numpy(params_m["Lx"], 0)) < 1e-15), "External potential fails  'periodic in x' test"
    #
    if "Ly" in params_m:
      assert(abs(func_numpy(0,0)-func_numpy(0,params_m["Ly"]))<1e-15), "External potential fails ' periodic in y' test"
    #
    print("External potential ", Vext)
    Vext_x = sympy.diff(Vext, "x")
    Vext_y = sympy.diff(Vext, "y")
    # grad
    func_numpy_x = lambdify((x, y), Vext_x, modules=[np], dummify=False)
 
    # alternative implementation - speed not important here
    # func_numexpr_x = lambdify((x, y), Vext_x, modules=[numexpr], dummify=False)
    #
    func_numpy_y = lambdify((x, y), Vext_y, modules=[np], dummify=False)
    # func_numexpr_y = lambdify((x, y), Vext_y, modules=[numexpr], dummify=False)
    #
    if "Ly" in params_m:

        def ext_grad(X):
            return np.array(
                [func_numpy_x(X[:, 0], X[:, 1]), func_numpy_y(X[:, 0], X[:, 1])]
            ).T

    else:

        def ext_grad(X):
            return np.array(func_numpy_x(X, 0))

    return ext_grad


###############
def get_ext_pot(params_m, params_n):
    # returns  an ext_pot function for use in weak form
    # returns None if non needed
    if "external" in params_m:  # external potential needed
        if type(params_m["external"]) == str:
            f = str_ext_pot_to_force(params_m)
        else:
            f = params_m["external"]
        if params_n["element_j"]=="Raviart-Thomas":
          V = VectorFunctionSpace( params_n["mesh"], "DG", params_n["degree"])
        else:
          V = params_n["trial_j"]
        m = V.ufl_domain()  # mesh
        # Interpolate the coordinates onto the nodes of W.
        X = interpolate(m.coordinates, V)
        X_values = X.dat.data_ro
        ext_pot = Function(V)
        if "no_y" in params_n and "Ly" in params_m:  # 2d
            ext_pot.dat.data[:, 0] = f(X_values)[:, 0]
            ext_pot.dat.data[:, 1] = f(X_values)[:, 1]
        else:
            ext_pot.dat.data[:] = f(X_values).flatten()
    else:  # no external potential
        print("No external potential")
        ext_pot = None
    #
    return ext_pot


#################
def convolution(params_m, params_n):
    # returns a function to compute the convolution rho_con= V*rho
    # returns None if not needed
    # TODO; more testing!!
    if (
        "pair" not in params_m or params_n["convolution_radius"] < 1e-15
    ):  # no convolution
        print("No convolution (pair potential)")
        rho_con = None
        return lambda x, y: 0, rho_con
    #
    if type(params_m["pair"]) == str:
        f = str_radial_pair_pot_to_force(params_m)
    else:
        f = params_m["pair"]
    # convolution needed
    t = time.perf_counter()
    print("  Creating convolution matrix...",end="")
    V = params_n["trial_rho"]
    m = V.ufl_domain()  # mesh
    W = VectorFunctionSpace(m, V.ufl_element())
    # Interpolate the coordinates onto the nodes of W.
    Xrow = interpolate(m.coordinates, params_n["trial_j"])
    Xcol = interpolate(m.coordinates, W)  # from rho
    Xrow_values = Xrow.dat.data_ro
    Xcol_values = Xcol.dat.data_ro
    nrow = Xrow_values.shape[0]
    ncol = Xcol_values.shape[0]
    #
    r = params_n["convolution_radius"]
    if "no_y" in params_n and "Ly" in params_m:  # 2d
        weight = pi * (r ** 2)
        lst1 = np.linspace(-1, 1, 3) * [[params_m["Lx"]]]
        lst2 = np.linspace(-1, 1, 3) * [[params_m["Ly"]]]
        l1, l2 = np.meshgrid(lst1, lst2, indexing="ij")
        lst = np.vstack([l1.flatten(), l2.flatten()]).T
    else:  # 1d
        weight = 2 * r
        lst = np.linspace(-1, 1, 3) * params_m["Lx"]
        Xrow_values = Xrow_values.reshape(nrow, 1)
        Xcol_values = Xcol_values.reshape(ncol, 1)
    #  tree structure to efficiently compute sparse distance matrices
    tree_row = cKDTree(Xrow_values)
    mtx = scipy.sparse.coo_matrix((nrow, ncol))
    unit_vec = 0
    for offset in lst:  # periodic copies
        tree_col = cKDTree(Xcol_values + offset)
        mtx1 = tree_row.sparse_distance_matrix(tree_col, r, output_type="coo_matrix")
        mtx1.eliminate_zeros()  # remove pairs with identical positions
        delta = Xrow_values[mtx1.row, :] - (Xcol_values[mtx1.col, :] + offset)
        extra_unit_vec = np.einsum("ij,i->ij", delta, 1 / mtx1.data)
        if type(unit_vec) == int:  # first run
            unit_vec = extra_unit_vec
        else:
            unit_vec = np.vstack([unit_vec, extra_unit_vec])
        mtx.row = np.append(mtx.row, mtx1.row)
        mtx.col = np.append(mtx.col, mtx1.col)
        mtx.data = np.append(mtx.data, mtx1.data)
    #
    nnz = mtx.getnnz(axis=0)  # number of non-zero entries
    # apply pair potential and normalise
    tmp = f(mtx.data) * weight / nnz[mtx.col]
    #
    rho_con = Function(params_n["trial_j"])
    if "no_y" in params_n and "Ly" in params_m:  # 2d
        mtx1.row = mtx.row
        mtx1.col = mtx.col  # copy of mtx
        mtx.data = tmp * unit_vec[:, 0]
        mtx1.data = tmp * unit_vec[:, 1]
        # convert to convolution matrix to spare csc format
        mtx.tocsc()
        mtx1.tocsc()

        def do_convolution(in_fn, out_fn):
            out_fn.dat.data[:, 0] = mtx * in_fn.dat.data[:]
            out_fn.dat.data[:, 1] = mtx1 * in_fn.dat.data[:]

    else:
        # convert to csc format
        mtx.data = tmp * unit_vec.flatten()
        mtx = mtx.tocsc()

        def do_convolution(in_fn, out_fn):
            out_fn.dat.data[:] = mtx * in_fn.dat.data[:]

    #
    #
    print("finished in {:.1f}".format(time.perf_counter() - t), "secs")
    #
    return do_convolution, rho_con


#################################
def periodic_distance(x0, x1, dimensions):
    # unused - implements a distance for periodic domain for use with pair potential
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))


#
##################################
def get_test_trial_space(params_m, params_n):

    # if params_n["degree"] == 0:
    #   print("Changing to degree=1 (piecewise constant fails ... todo)")
    #   params_n["degree"]=1      
    if params_n["noise_degree"] == 0 and params_n["element_noise"]=="CG":
       print("Changing to noise_degree=1 (for CG)")
       params_n["noise_degree"]=1 
    # create mesh and test/trial spaces for rho and j as entries in params_n
    if "no_y" in params_n and "Ly" in params_m:  # 2d
        # Set-up mesh
        print("Periodic rectangular mesh of size ", params_n["no_x"],"x",params_n["no_y"])
        mesh = PeriodicRectangleMesh(
            params_n["no_x"],
            params_n["no_y"],
            params_m["Lx"],
            params_m["Ly"],
            direction="both",
        )
    else:  # 1d
        print("Periodic interval mesh of size ", params_n["no_x"])
        mesh = PeriodicIntervalMesh(params_n["no_x"], params_m["Lx"])
    #
    degree = params_n["degree"]
    print(
        "Elements: (rho,j)=(",
        params_n["element_rho"],
        ", ",
        params_n["element_j"],
        ") of degree",
        degree,
        "; noise=",
        params_n["element_noise"],
        "of degree",
        params_n["noise_degree"],
    )
    params_n["mesh"] = mesh
    params_n["trial_rho"] = FunctionSpace(mesh, params_n["element_rho"], degree)
    params_n["test_rho"] = FunctionSpace(mesh, params_n["element_rho"], degree)
    # vector finite-elements for momentum density j (even in 1d - easier to translate to 2d)
    if params_n["element_j"]=="Raviart-Thomas" and "Ly" not in params_m:
      print("Raviart-Thomas not available in 1d.")
      exit(1)
    if params_n["element_j"]=="Raviart-Thomas" and "Ly" in params_m:
       params_n["trial_j"] = FunctionSpace(mesh, params_n["element_j"], degree+1)
       params_n["test_j"] = FunctionSpace(mesh, params_n["element_j"], degree+1)
    else:
      params_n["trial_j"] = VectorFunctionSpace(mesh, params_n["element_j"], degree)
      params_n["test_j"] = VectorFunctionSpace(mesh, params_n["element_j"], degree)
    params_n["noise_space"] = VectorFunctionSpace(
        mesh, params_n["element_noise"], params_n["noise_degree"]
    )


#############################
def set_up_test(params_m, params_n):
    # choose an eigenpair according to params_n
    # return initial data in trial space (as defined in params_n)
    # 1d only
    nn = params_n["initial_data_number"]
    y = params_n["initial_data_param"]
    lam, eig_fn_rho, eig_fn_rho1, eig_fn_j = efun(nn, y, params_m)
    # initial fields
    (x,) = SpatialCoordinate(params_n["mesh"])
    #
    if type(lam) == real:
        params_m["initial_rho"] = lambda x: eig_fn_rho(x) + params_m["mass"]
        params_m["initial_j"] = lambda x: [eig_fn_j(x) + params_m["wind"]]
    else:
        params_m["initial_rho"] = lambda x: eig_fn_rho(x) + params_m["mass"]
        params_m["initial_j"] = lambda x: [eig_fn_j(x) + params_m["wind"]]
    # density rho
    params_n["rho0"] = Function(params_n["trial_rho"]).interpolate(
        params_m["initial_rho"](x)
    )
    # momentum density j
    tmp_j = as_vector(params_m["initial_j"](x))
    params_n["j0"] = Function(params_n["trial_j"]).interpolate(tmp_j)
    # exact solutions
    params_m["exact_rho"] = exact_rho(lam, params_m["mass"], eig_fn_rho, eig_fn_rho1)
    params_m["exact_j"] = exact_j(lam, params_m["wind"], eig_fn_j)
    j0 = Function(params_n["trial_j"])
    rho0 = Function(params_n["trial_rho"])
    rho0.assign(params_n["rho0"])
    j0.assign(params_n["j0"])
    return rho0, j0


#
def set_up_1d(params_m, params_n):
    # set-up 1d (no exact solution)
    mesh = params_n["mesh"]
    # initial fields
    (x,) = SpatialCoordinate(mesh)
    # density
    rho0 = Function(params_n["trial_rho"]).interpolate(params_m["initial_rho"](x))
    total_mass = assemble(rho0 * dx)
    def rho0_scaled(x): return params_m["mass"]*params_m["initial_rho"](x)/total_mass
    rho0 = Function(params_n["trial_rho"]).interpolate(rho0_scaled(x))
    # momentum density
    tmp_j = as_vector(params_m["initial_j"](x))
    j0 = Function(params_n["trial_j"]).interpolate(tmp_j)
    return rho0, j0
#
def set_up_2d(params_m, params_n):
    # set-up simple 2d example (no exact solution)
    mesh = params_n["mesh"]
    # initial fields
    x, y = SpatialCoordinate(mesh)
    # density
    rho0 = Function(params_n["trial_rho"]).interpolate(params_m["initial_rho"](x, y))
    total_mass = assemble(rho0 * dx)
    def rho0_scaled(x,y): return params_m["mass"]*params_m["initial_rho"](x, y)/total_mass
    # momentum density
    rho0 = Function(params_n["trial_rho"]).interpolate(rho0_scaled(x, y))
    
    tmp_j = as_vector(params_m["initial_j"](x, y))
    j0 = Function(params_n["trial_j"]).project(tmp_j)
    return rho0, j0






# def smoothclamp(x, mi, mx):
#  return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )
def smoothclamp(x, mi, mx):
 return  (lambda t: conditional(t < 0 , 0, conditional( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

####################################
def get_form_dict(rho, rho0, j, j0, phi, psi, params_m, params_n):
    # Lots of common defintion used for weak forms
    X = {}
    X["gamma"] = params_m["gamma"]
    sigma_o_root_N = params_m["sigma"] / sqrt(params_m["N"])
    cc = params_m["sigma"] ** 2 / params_m["gamma"] / 2
    #
    if "c_to_one" in params_m:
      if "no_print" not in params_n:
          print("Changed param_c to one (to fit with Federico's calculation)")
      cc = 1
    assert(cc>0), "Must have c>0 (i.e., sigma>0)"
    #
    half_root_c = (0.5 * sqrt(cc))
    one_o_root_c = (1/sqrt(cc))
    #
    if "no_print" not in params_n:
      print("epsilon={:.3e}".format(params_m["epsilon"]),
        "; c={:.3e}".format(cc),
        "; gamma={:.3e}".format(params_m["gamma"]),
        "; sigma/root(N)={:.3e}".format(params_m["sigma"] / sqrt(params_m["N"])),
         "; final time={:.1e}".format(params_m["final_time"] )
    )
    # phi regulariation
    X["cutoff_phi"]=1
    X["cutoff"]=conditional(rho0>0,1.,0.)
    if params_n["tau"]>0:
      X["cutoff_phi"]=smoothclamp(rho0,params_n["tau"]/2,params_n["tau"])
    #
    # To turn off noise and potential
    if params_n["rho_thresh"]>0: # more agressive turn-off
      print("Redefine cut-off")
      X["cutoff"]=smoothclamp(rho0,0,params_n["rho_thresh"])
    ########################################
    # Elements of the weak form
    X["a_rho"] = cc*rho * phi
    X["L_rho"] = cc*rho0 * phi
    X["a_j"] = dot(j, psi)
    X["L_j"] = dot(j0, psi)
    ################################
    # Flux
    # normal
    n = FacetNormal(params_n["mesh"])
    # implicit
    hj = dot(avg(j), n('+')) + half_root_c  * jump(rho)
    hrho = (avg(rho) + one_o_root_c * avg(dot(j, n)))
    #explicit
    hj0 = dot(avg(j0), n('+')) +half_root_c * jump(rho0)
    hrho0= (avg(rho0) + one_o_root_c * avg(dot(j0, n)))
    #
    X["a_flux_rho"] = cc*jump(phi)* hj 
    X["L_flux_rho"] =  cc*jump(phi) * hj0   
    X["a_flux_j"] = cc * 2 * hrho * avg(dot(psi, n))
    X["L_flux_j"] = cc * 2 * hrho0 * avg(dot(psi, n))
    #
    X["a_flux_both"] = X["a_flux_rho"] + X["a_flux_j"]
    X["L_flux_both"] = X["L_flux_rho"] + X["L_flux_j"]
    #
    #
    X["ad"] = (cc*dot(grad(phi), j) + cc * div(psi) * rho )*dx-  X["a_flux_both"]*dS
    #####################
    # Potential
    if "no_print" not in params_n:
      print("Setting up potential")
    if "pot" in params_n:
      X["do_convolution"], X["rho_con"] = convolution(params_m, params_n)
    else:
      X["rho_con"] = None
    #
    if "inter_pair" in params_m:
      params_m1=params_m.copy()
      params_m1["pair"]=params_m["inter_pair"]
      X["do_convolution_inter"], X["rho_con_inter"] = convolution(params_m1, params_n)
    #
    if "pot" in params_n:
      ext_pot = get_ext_pot(params_m, params_n)
    else:
      ext_pot=None
    #
    if not (X["rho_con"] is None) :
        X["L_conv"] = dot(rho0 * X["rho_con"], psi)
    else:
        X["L_conv"] = 0
    # inter
    if "inter_pair" in params_m:
      print("Added inter-particle pair to weak form")
      X["L_conv"]=X["L_conv"]+dot(rho0 * X["rho_con_inter"], psi)
    #
    if not (ext_pot is None):
        X["L_ext"] = dot(rho0*ext_pot, psi)*X["cutoff"]
    else:
        X["L_ext"] = 0
    # both potential terms
    X["L_pot"]=X["L_conv"]+X["L_ext"]
    #
    #########################
    # Noise
    if "no_print" not in params_n:
      print("Setting-up noise")
    xi, X["sample"] = get_noise(params_m, params_n)
    #
    if not (xi is None):
        X["L_noise"] = sigma_o_root_N *sqrt(rho0*X["cutoff"]*X["cutoff_phi"] ) * dot(psi, xi)
        X["L_noise2"] = sigma_o_root_N *sqrt(rho0*X["cutoff"] ) * dot(psi, xi)
    else:
        if "no_print" not in params_n:
          print("No noise")
          X["L_noise"] = 0
    ################################
    # Reaction
    if "no_print" not in params_n:
      print("Setting-up reaction")
    if "MD_rate" in params_m:
      if "Ly" in params_m:
        react_V=np.pi*params_m["reaction_radius"]**2
      else:
        react_V=2*params_m["reaction_radius"]
      params_m["reaction_rate"]=params_m["MD_rate"]*react_V *params_m["N"]
      # print("Reaction rate", params_m["reaction_rate"], " from MD rate ",params_m["MD_rate"])
    else:
      params_m["reaction_rate"]=0.
    # 
    params_n["reaction_rate_dt"]=params_m["reaction_rate"]*params_n["dt"]
    #
    reaction_field, X["set_ext_reaction"] = get_ext_reaction(params_m, params_n)
    my_case=0
    thresh=1.4e-2#1.2e-2
    F1a=conditional(rho0>thresh,rho0,0)
    F1=rho0
    F2a=conditional(reaction_field>thresh,reaction_field,0)
    F2=reaction_field
    if my_case==1:
      X["rf_sink"]=   -((exp(-params_n["reaction_rate_dt"]*F1)-1)*F2  *phi)
      X["rf_source"]=  ((exp(-params_n["reaction_rate_dt"]*F2)-1)*F1  
    *phi)
    else:
      X["rf_sink"]=   ((exp(params_n["reaction_rate_dt"]*F2 -0.5*params_n["reaction_rate_dt"]**2*F2*F1a)-1)*F1a  *phi)
      X["rf_source"]= -((exp(params_n["reaction_rate_dt"]*F1 -0.5*params_n["reaction_rate_dt"]**2*F2a*F1)-1)*F2a  *phi)
    #
    # X["rf_sink"]=   params_n["reaction_rate_dt"]*reaction_field*rho0*phi
    # X["rf_source"]=  -X["rf_sink"]
    #   X["L_rho"] = cc*rho0 * phi
    if params_m["reaction_rate"]>0:
      if params_m["source"]:#source
        X["reaction"] = cc*X["rf_source"]
      else:# sink
        X["reaction"] = cc*X["rf_sink"]
    else: # no reaction
        X["reaction"] = 0
    params_n["no_print"]=True
    ######
    # SIPG
    # SIPG form for heat equation
    #
    if "no_print" not in params_n:
      print("Setting-up heat equation")
    if params_n["SIPG_D"]>0:
      reg_SIPG=params_n["no_x"]/1*params_n["SIPG_reg"]
      #params_n["SIPG_D"]=c/params_m["gamma"]
      cc_SIPG=1 
      a_flux_SIPG= cc *(avg(rho)*avg(dot(psi,n)))+(params_n["SIPG_D"])*( dot( avg(grad(rho)) ,avg(n*phi)) - reg_SIPG*jump(rho)*jump(phi))
    
      a1_SIPG=lambda dt: dt*(dot(j,psi) -params_n["SIPG_D"] *rho*div(psi))*dx +dt*params_n["SIPG_D"] *avg(rho)*avg(dot(psi,n))*dS
  
      a2_SIPG=lambda dt : ((rho*phi -dt*dot(j,grad(phi))))*dx+dt*(params_n["SIPG_D"])*( dot( avg(grad(rho)) ,avg(n*phi)) - reg_SIPG*jump(rho)*jump(phi))*dS
    
      X["a_SIPG"]=lambda dt : a1_SIPG(dt) + a2_SIPG(dt)
      X["a_flux_SIPG"]= a_flux_SIPG 
      X["L_SIPG"]=lambda dt : (rho0*phi)*dx
    #  
    return X
###################################
def weak_form_em(params_m, params_n, rho0, j0):
    # create weak form for Euler-Maruyama time-stepping
    # return function to move one step
    # time interval and time step
    params_n["dt"] = params_m["final_time"] / params_n["no_t"]
    dtc = params_n["dt"]
    dt_sqrt = sqrt(params_n["dt"])
    #
    V = params_n["trial_rho"] * params_n["trial_j"]
    W = params_n["test_rho"] * params_n["test_j"]
    rho, j = TrialFunctions(V)
    phi, psi = TestFunctions(W)
    # Set up weak form
    params_n["pot"]=True
    X = get_form_dict(rho, rho0, j, j0, phi, psi, params_m, params_n)
    # 
    print("Time stepping: Euler-Maruyama method (implicit)")
    #
    # Define weak form
    #
    a = (X["a_rho"] + (X["cutoff_phi"] + dtc*params_m["gamma"]) * X["a_j"])*dx-dtc*X["ad"]
    # 
    L = (
            X["L_rho"] +X["reaction"]
            + X["cutoff_phi"] *X["L_j"]
            + dtc * ( - X["L_pot"])
            + dt_sqrt * X["L_noise"]
        ) * dx  
    #
    u = Function(V)
    print("Setting up variational problem")
    prob1 = LinearVariationalProblem(a, L, u)
    solv1 = LinearVariationalSolver(
        prob1, solver_parameters=params_n["solver_parameters"]  )
    #
    # heat equation (for smoothing)
    if params_n["SIPG_D"]>0:
      a_he = X["a_SIPG"](dtc)
      L_he = X["L_SIPG"](dtc)
      u_ = Function(V)
      prob_he = LinearVariationalProblem(a_he, L_he, u_)
      solv_he = LinearVariationalSolver(prob_he, solver_parameters=params_n["solver_parameters"])
    #limiter=VertexBasedLimiter(params_n["trial_rho"])
    #limited kills oscillations between conservation law is lost
    def one_step(tmp):
        X["do_convolution"](rho0, X["rho_con"])
        X["sample"]()
        X["set_ext_reaction"](tmp)
        solv1.solve()
        j0.assign(u.sub(1))
        rho0.assign(u.sub(0))
        #limiter.apply(rho0)      #
        if params_n["SIPG_D"]>0:
               solv_he.solve()
               rho0.assign(u_.sub(0))
    #
    return one_step, params_n["dt"]
# end weak_form_em

# end
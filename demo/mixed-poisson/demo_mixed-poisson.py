# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Mixed Poisson problem with a Schur complement preconditioner

# A classic example of a block-structured linear system arises from the finite
# element discretisation of the mixed Poisson problem
#
# $$
# \begin{align}
# q - \nabla{p} &= 0 &\textrm{in}\ \Omega, \\
# \mathrm{div}(q) &= -f &\textrm{in}\ \Omega, \\
# p &= 0 &\textrm{on}\ \Gamma_D, \\
# q \cdot n &= g &\textrm{on}\ \Gamma_N,
# \end{align}
# $$
#
# for a domain $\Omega \subset \mathbb{R}^2$ with boundary
# $\partial \Omega = \Gamma_D \cup \Gamma_N$.
# In this example we solve for the same boundary conditions used in the DOLFINx's standard Poisson
# equation [demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_poisson.html),
# namely:
#
# - $\Omega = [0,2] \times [0,1]$.
# - $\Gamma_{D} = \{(0, y) \cup (2, y) \subset \partial \Omega\}$.
# - $\Gamma_{N} = \{(x, 0) \cup (x, 1) \subset \partial \Omega\}$.
# - $g = \sin(5x)$.
# - $f = 10\exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)$.
#
# In weak form we seek a vector-valued flux $q_h \in Q_h$ and scalar-valued pressure $p_h \in P_h$,
# such that
#
# $$
# \begin{align}
# \left( q_h, \tilde{q} \right) + \left( p_h, \mathrm{div}(\tilde{q}) \right)
#   &= 0 \quad &\forall \tilde{q} \in Q_h,\\
# \left( \mathrm{div}(q_h), \tilde{p} \right)
#   &= \left( -f_h, \tilde{p} \right) \quad &\forall \tilde{p} \in P_h.
# \end{align}
# $$
#
# where $\left( \cdot, \cdot \right)$ denotes the usual $L^2$ inner product on
# the finite element mesh and $f$ a known forcing term. In discrete block form
# this can be written as a saddle point linear system with unknown vectors of
# finite element coefficients $q$ and $p$
#
# $$
# \begin{align}
# \begin{bmatrix}
# A & B^T \\
# B & O
# \end{bmatrix}
# \begin{bmatrix}
# q \\ p
# \end{bmatrix}
# &=
# \begin{bmatrix}
# 0 \\ f
# \end{bmatrix},
# \end{align}
# $$
#
# where $A$ is a square matrix arising from the bilinear form $(q_h, \tilde{q})$,
# $B$ and $B^T$ are non-square matrices arising from the bilinear forms
# $(\mathrm{div}(q_h), \tilde{p})$ and $(p_h, \mathrm{div}(\tilde{q}))$
# respectively, $O$ is a square matrix of zeros, $0$ is a vector of zeros and $f$
# is a vector arising from the linear form $-(f_h, \tilde{p})$. For the ease of notation,
# we rewrite the above system in a compact form
#
# $$
# \begin{align}
# K x &= b,
# \end{align}
# $$
#
# $$
# \begin{align}
# K &=
# \begin{bmatrix}
# A & B^T \\
# B & O
# \end{bmatrix}, &
# x &=
# \begin{bmatrix}
# q \\ p
# \end{bmatrix}, &
# b &=
# \begin{bmatrix}
# 0 \\ f
# \end{bmatrix}.
# \end{align}
# $$
#
# The block-structured matrix $K$ and vector $b$ can be assembled using the
# standard code shown below. We define a mesh consisting of quadrilateral
# cells. For the flux space $V_h$ we choose Brezzi-Douglas-Marini elements of
# first-order, and for the pressure space $Q_h$ discontinuous Lagrange elements
# of zeroth-order.

# +
import pathlib

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from basix.ufl import element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_matrix_block,
    assemble_vector_block,
)
from fenicsx_pctools.mat import create_splittable_matrix_block
from fenicsx_pctools.utils import vec_to_functions
from ufl import (
    CellDiameter,
    FacetNormal,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    avg,
    div,
    exp,
    grad,
    inner,
    jump,
)

# Define mesh
nx, ny = 1024, 1024
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([2.0, 1.0])],
    [nx, ny],
    mesh.CellType.quadrilateral,
)

k = 1
Q_el = element("BDMCF", domain.basix_cell(), k)
P_el = element("DG", domain.basix_cell(), k - 1)
Q = fem.functionspace(domain, Q_el)
P = fem.functionspace(domain, P_el)

q = TrialFunction(Q)
q_t = TestFunction(Q)

p = TrialFunction(P)
p_t = TestFunction(P)


def boundary_top(x):
    return np.isclose(x[1], 1.0)


def boundary_bottom(x):
    return np.isclose(x[1], 0.0)


fdim = domain.topology.dim - 1
facets_top = mesh.locate_entities_boundary(domain, fdim, boundary_top)
facets_bottom = mesh.locate_entities_boundary(domain, fdim, boundary_bottom)

blocked_dofs_up = fem.locate_dofs_topological(Q, fdim, facets_top)
blocked_dofs_down = fem.locate_dofs_topological(Q, fdim, facets_bottom)


def f1(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = np.sin(5 * x[0])
    return values


def f2(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = -np.sin(5 * x[0])
    return values


f_h1 = fem.Function(Q)
f_h1.interpolate(f1)
f_h2 = fem.Function(Q)
f_h2.interpolate(f2)

bc_up = fem.dirichletbc(f_h1, blocked_dofs_up)
bc_down = fem.dirichletbc(f_h2, blocked_dofs_down)
bcs = [bc_up, bc_down]

x = SpatialCoordinate(domain)
f = 10.0 * exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)

dx = Measure("dx", domain)
a = [[inner(q, q_t) * dx, inner(p, div(q_t)) * dx], [inner(div(q), p_t) * dx, None]]
L = [inner(fem.Constant(domain, (0.0, 0.0)), q_t) * dx, -inner(f, p_t) * dx]
a_dolfinx = fem.form(a)
L_dolfinx = fem.form(L)

K = assemble_matrix_block(a_dolfinx, bcs)
K.assemble()

b = assemble_vector_block(L_dolfinx, a_dolfinx, bcs)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# -
# FEniCSx-pctools method `create_splittable_matrix_block` takes the DOLFINx
# assembled matrix `K` and the associated block UFL form `a` and returns a
# PETSc Mat of type `"python"` with the necessary functionality to apply PETSc
# block preconditioning strategies.

# +
K_splittable = create_splittable_matrix_block(K, a)

# -
# We now describe and specify an upper-diagonal Schur complement preconditioner
# using PETSc. Writing the $LDU$ decomposition of $K$ gives
#
# $$
# K = LDU =
# \begin{bmatrix}
# I & 0 \\
# BA^{-1} & I
# \end{bmatrix}
# \begin{bmatrix}
# A & 0 \\
# 0 & S
# \end{bmatrix}
# \begin{bmatrix}
# I & A^{-1}B^T \\
# 0 & I
# \end{bmatrix}.
# $$
#
# where $S = -B A^{-1} B^T$ is known as the Schur complement. Choosing to use
# the inverse of the diagonal $D$ and upper $U$ components as a preconditioner
#
# $$
# P_{\mathrm{upper}} = DU =
# \begin{bmatrix}
# A & 0 \\
# 0 & S
# \end{bmatrix}
# \begin{bmatrix}
# I & A^{-1}B^T \\
# 0 & I
# \end{bmatrix},
# $$
#
# leads to the following *upper* Schur complement *left* preconditioned block
# system of equations
#
# $$
# P_{\mathrm{upper}}^{-1} K x = P_{\mathrm{upper}}^{-1} b.
# $$
#
# Use of GMRES and upper Schur complement preconditioning can be specified
# using the PETSc options shown below. We first tell PETSc that we would like
# to use GMRES as an outer solver with the default left preconditioning.

# +
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(K_splittable)

options = PETSc.Options()
options.prefixPush("mp_")
options["ksp_type"] = "gmres"
options["ksp_rtol"] = 1e-8
options["ksp_monitor_true_residual"] = ""

# -
# The specification of Schur complement preconditioning requires a PETSc preconditioner of type
# `"fieldsplit"`. However, since the provided preconditioning operator is a splittable matrix,
# we have to wrap the preconditioner using a dedicated class `"fenicsx_pctools.pc.WrappedPC"`.

# +
options["pc_type"] = "python"
options["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"

# -
# This additional wrapper enables the interaction of the preconditioner with the wrapped
# PETSc matrix, and unlocks the possibility to define the splitting scheme using block
# indices as runtime options. The second feature is not of a practical importance for this
# simple $2 \times 2$ block system, but it can be very useful when we aim to test different
# solver configurations with nested splitting schemes without necessarily changing the code
# (see [Solver configuration at runtime](demo_solver-config.md)).
#
# Now we can inform PETSc about our intention to use
#
# $$
# P_{\mathrm{upper}}^{-1} =
# \begin{bmatrix}
# I & -A^{-1}B^T \\
# 0 & I
# \end{bmatrix}
# \begin{bmatrix}
# A^{-1} & 0 \\
# 0 & S^{-1}
# \end{bmatrix}
# $$
# in the left preconditioned algebraic system.

# +
options.prefixPush("wrapped_")
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_0_fields"] = "0"
options["pc_fieldsplit_1_fields"] = "1"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_fact_type"] = "upper"
options["pc_fieldsplit_schur_precondition"] = "user"

# -
# The last option says that $S^{-1}$ will be specified by the user. We will explain this shortly.
#
# In the general case $S$ is a dense matrix that cannot be stored explicitly,
# let alone inverted. To avoid this, we suppose the existence of a 'good'
# approximate action for both $A^{-1} \approx \tilde{A}^{-1}$ and $S^{-1}
# \approx \tilde{S}^{-1}$, i.e. we substitute
#
# $$
# \tilde{P}_{\mathrm{upper}}^{-1} =
# \begin{bmatrix}
# I & -\tilde{A}^{-1}B^T \\
# 0 & I
# \end{bmatrix}
# \begin{bmatrix}
# \tilde{A}^{-1} & 0 \\
# 0 & \tilde{S}^{-1}
# \end{bmatrix},
# $$
#
# for $P_{\mathrm{upper}}^{-1}$ where the tilde $(\tilde{\cdot})$ denotes an
# approximate (inexact) inverse. To actually compute $\tilde{P}_{\mathrm{upper}}^{-1}$
# we still must specify the form of both $\tilde{A}^{-1}$ and $\tilde{S}^{-1}$.
#
# One reasonable choice is to take $\tilde{A}^{-1}$ as a single application of
# a block Jacobi preconditioned inverse mass matrix on the finite element flux
# space $Q_h$. We note that the operator $\tilde{A} = A$ and by default PETSc
# re-uses the upper left block $A$ in $K$. This can be specified using the
# following code.

# +
options.prefixPush("fieldsplit_0_")
options["ksp_type"] = "preonly"
options["pc_type"] = "bjacobi"
options.prefixPop()  # fieldsplit_0_

# -
# For $\tilde{S}^{-1}$ we take a single application of algebraic multigrid
# preconditioned discontinuous Galerkin approximation of the Laplacian on the
# finite element pressure space $P_h$. We setup a class `SchurInv` with a
# method `apply` that will apply the approximate inverse of the discontinuous
# Galerkin Laplacian operator `S` to the vector `x`. We then tell PETSc to use
# this method when it needs the action of $\tilde{S}^{-1}$.


# +
def boundary_left_and_right(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 2.0))


facets_left_and_right = mesh.locate_entities_boundary(
    domain, domain.topology.dim - 1, boundary_left_and_right
)
sort = np.argsort(facets_left_and_right)
facet_tag = mesh.meshtags(
    domain,
    domain.topology.dim - 1,
    facets_left_and_right[sort],
    np.ones_like(facets_left_and_right),
)

dS = Measure("dS", domain)
ds = Measure("ds", domain, subdomain_data=facet_tag)

n = FacetNormal(domain)
alpha = fem.Constant(domain, 4.0)
gamma = fem.Constant(domain, 8.0)
h = CellDiameter(domain)

s = -(
    inner(grad(p), grad(p_t)) * dx
    - inner(avg(grad(p_t)), jump(p, n)) * dS
    - inner(jump(p, n), avg(grad(p_t))) * dS
    + (alpha / avg(h)) * inner(jump(p, n), jump(p_t, n)) * dS
    - inner(grad(p), p_t * n) * ds(1)
    - inner(p * n, grad(p_t)) * ds(1)
    + (gamma / h) * p * p_t * ds(1)
)

S = assemble_matrix(fem.form(s))
S.assemble()


class SchurInv:
    def setUp(self, pc):
        self.ksp = PETSc.KSP().create()
        self.ksp.setOptionsPrefix(pc.getOptionsPrefix() + "SchurInv_")
        self.ksp.setOperators(S)
        self.ksp.setFromOptions()

    def apply(self, pc, x, y):
        self.ksp.solve(x, y)


options.prefixPush("fieldsplit_1_")
options["ksp_type"] = "preonly"
options["pc_type"] = "python"
options["pc_python_type"] = __name__ + ".SchurInv"
options.prefixPush("SchurInv_")
options["ksp_type"] = "preonly"
options["pc_type"] = "hypre"
options.prefixPop()  # SchurInv_
options.prefixPop()  # fieldsplit_1_

options.prefixPop()  # wrapped_
options.prefixPop()  # mp_

# -
#
# Finally, we set all of the options on the PETSc objects and solve. This
# solver setup gives a nearly mesh independent number of GMRES iterations (~20)
# tested up to a mesh size of $1024 \times 1024$.

# +
solver.setOptionsPrefix("mp_")
solver.setFromOptions()

u_h = fem.petsc.create_vector_block(fem.form(L))
solver.solve(b, u_h)

q_h = fem.Function(Q)
p_h = fem.Function(P)
vec_to_functions(u_h, [q_h, p_h])

Q_out_el = element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
Q_out = fem.functionspace(domain, Q_out_el)

q_h_out = fem.Function(Q_out)
q_h_out.interpolate(q_h)

outdir = pathlib.Path(__file__).resolve().parent.joinpath("output")

with io.XDMFFile(MPI.COMM_WORLD, outdir.joinpath("q_h.xdmf"), "w") as handle:
    handle.write_mesh(domain)
    handle.write_function(q_h_out)

with io.XDMFFile(MPI.COMM_WORLD, outdir.joinpath("p_h.xdmf"), "w") as handle:
    handle.write_mesh(domain)
    handle.write_function(p_h)

solver.destroy()
K_splittable.destroy()
K.destroy()
S.destroy()
b.destroy()
# -

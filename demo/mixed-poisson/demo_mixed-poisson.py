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

import numpy as np
import pathlib

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_matrix_block
from fenicsx_pctools.utils import vec_to_functions
from fenicsx_pctools.mat import create_splittable_matrix_block
from ufl import (
    CellDiameter,
    FacetNormal,
    FiniteElement,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorElement,
    avg,
    div,
    exp,
    grad,
    inner,
    jump,
)

from mpi4py import MPI
from petsc4py import PETSc

# Define mesh
nx, ny = 128, 128
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
    [nx, ny],
    mesh.CellType.quadrilateral,
)

k = 1
Q_el = FiniteElement("BDMCF", domain.ufl_cell(), k)
P_el = FiniteElement("DG", domain.ufl_cell(), k - 1)
Q = fem.FunctionSpace(domain, Q_el)
P = fem.FunctionSpace(domain, P_el)

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

A = create_matrix_block(a_dolfinx)
assemble_matrix_block(A, a_dolfinx, bcs)
A.assemble()

A_splittable = create_splittable_matrix_block(A, a)
A_splittable.setOptionsPrefix("mp_")

b = assemble_vector_block(L_dolfinx, a_dolfinx, bcs)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# Interior penalty DG preconditioner
def boundary_left_and_right(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))


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
a_p_11 = (
    -inner(grad(p), grad(p_t)) * dx
    + inner(avg(grad(p_t)), jump(p, n)) * dS
    + inner(jump(p, n), avg(grad(p_t))) * dS
    - (alpha / avg(h)) * inner(jump(p, n), jump(p_t, n)) * dS
    + inner(grad(p), p_t * n) * ds(1)
    + inner(p * n, grad(p_t)) * ds(1)
    - (gamma / h) * p * p_t * ds(1)
)
a_p = [[inner(q, q_t) * dx, None], [None, a_p_11]]
a_p_dolfinx = fem.form(a_p)

A_P = fem.petsc.create_matrix_block(a_p_dolfinx)
fem.petsc.assemble_matrix_block(A_P, a_p_dolfinx, bcs)
A_P.assemble()

A_P_splittable = create_splittable_matrix_block(A_P, a_p)
A_P_splittable.setOptionsPrefix("mp_")

solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A_splittable, A_P_splittable)

options = PETSc.Options()
options.prefixPush("mp_")
options["ksp_type"] = "gmres"
options["ksp_rtol"] = 1e-8
options["ksp_monitor_true_residual"] = ""
options["pc_type"] = "python"
options["pc_python_type"] = "fenicsx_pctools.WrappedPC"

options.prefixPush("wrapped_")
options["pc_type"] = "fieldsplit"
options["pc_fieldsplit_type"] = "schur"
options["pc_fieldsplit_schur_fact_type"] = "upper"
options["pc_fieldsplit_schur_precondition"] = "user"
options["pc_fieldsplit_0_fields"] = "0"
options["pc_fieldsplit_1_fields"] = "1"

options.prefixPush("fieldsplit_0_")
options["ksp_type"] = "preonly"
options["pc_type"] = "bjacobi"
options.prefixPop()  # fieldsplit_0_

options.prefixPush("fieldsplit_1_")
options["ksp_type"] = "preonly"
options["pc_type"] = "hypre"
options.prefixPop()  # fieldsplit_1_

options.prefixPop()  # wrapped_
options.prefixPop()  # mp_

solver.setOptionsPrefix("mp_")
solver.setFromOptions()

u_h = fem.petsc.create_vector_block(fem.form(L))
solver.solve(b, u_h)

q_h = fem.Function(Q)
p_h = fem.Function(P)
vec_to_functions(u_h, [q_h, p_h])

Q_out_el = VectorElement("Lagrange", domain.ufl_cell(), 1)
Q_out = fem.FunctionSpace(domain, Q_out_el)

q_h_out = fem.Function(Q_out)
q_h_out.interpolate(q_h)

outdir = pathlib.Path(__file__).resolve().parent.joinpath("output")

with io.XDMFFile(MPI.COMM_WORLD, outdir.joinpath("q_h.xdmf"), "w") as handle:
    handle.write_mesh(domain)
    handle.write_function(q_h_out)

with io.XDMFFile(MPI.COMM_WORLD, outdir.joinpath("p_h.xdmf"), "w") as handle:
    handle.write_mesh(domain)
    handle.write_function(p_h)

PETSc.garbage_cleanup()

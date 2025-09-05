# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # (Paper) Mixed Poisson problem with a Schur complement preconditioner

# This simplified example was shown in the accompanying paper to FEniCSx-pctools,
# see it on [arXiv](https://arxiv.org/abs/2402.02523).

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from basix.ufl import element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from fenicsx_pctools.mat import create_splittable_matrix_block
from ufl import (
    CellDiameter,
    FacetNormal,
    Measure,
    TestFunction,
    TrialFunction,
    ZeroBaseForm,
    avg,
    div,
    dS,
    ds,
    grad,
    inner,
    jump,
)

domain = mesh.create_unit_square(MPI.COMM_WORLD, 1024, 1024, mesh.CellType.quadrilateral)

k = 1
Q_el = element("BDMCF", domain.basix_cell(), k)
P_el = element("DG", domain.basix_cell(), k - 1)
Q = fem.functionspace(domain, Q_el)
P = fem.functionspace(domain, P_el)

q = TrialFunction(Q)
q_t = TestFunction(Q)

p = TrialFunction(P)
p_t = TestFunction(P)

f = fem.Function(P)
rng = np.random.default_rng()
f.x.array[:] = rng.uniform(size=f.x.array.shape)

dx = Measure("dx", domain)
a = [[inner(q, q_t) * dx, inner(p, div(q_t)) * dx], [inner(div(q), p_t) * dx, None]]
L = [ZeroBaseForm((q_t,)), -inner(f, p_t) * dx]
a_dolfinx = fem.form(a)
L_dolfinx = fem.form(L)

K = assemble_matrix(a_dolfinx, kind="mpi")
K.assemble()

b = assemble_vector(L_dolfinx, kind="mpi")
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

K_splittable = create_splittable_matrix_block(K, a)
K_splittable.setOptionsPrefix("mp_")

solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOptionsPrefix("mp_")
solver.setOperators(K_splittable)

options = PETSc.Options()
options.prefixPush("mp_")
options["ksp_type"] = "gmres"
options["ksp_rtol"] = 1e-8
options["ksp_monitor_true_residual"] = ""
options["pc_type"] = "python"
options["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"

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

n = FacetNormal(domain)
alpha = fem.Constant(domain, 4.0)
gamma = fem.Constant(domain, 8.0)
h = CellDiameter(domain)

s = -(
    inner(grad(p), grad(p_t)) * dx
    - inner(avg(grad(p_t)), jump(p, n)) * dS
    - inner(jump(p, n), avg(grad(p_t))) * dS
    + (alpha / avg(h)) * inner(jump(p, n), jump(p_t, n)) * dS
    - inner(grad(p), p_t * n) * ds
    - inner(p * n, grad(p_t)) * ds
    + (gamma / h) * p * p_t * ds
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

solver.setFromOptions()

x = fem.petsc.create_vector(L_dolfinx, kind="mpi")
solver.solve(b, x)

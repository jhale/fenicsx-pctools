# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Solver configuration at runtime

# This simple demo emphasizes the most powerful feature of FEniCSx-pctools.
# In particular, it illustrates the ability to change the solver configuration at runtime
# independently of the model formulation. We base the presentation on an elementary system
# of algebraic equations in order to keep the focus on the selected feature.

# ## Problem definition

# In what follows, we will solve a system of algebraic equations $A x = b$, where

# ```{math}
#    \begin{align}
#    A &=
#    \begin{bmatrix}
#       A_{00} & A_{01} & A_{02} \\
#       A_{10} & A_{11} & A_{12} \\
#       A_{20} & A_{21} & A_{22}
#    \end{bmatrix}
#    \end{align}
# ```

# is the **block $3 \times 3$ matrix** representation of a linear operator,
# $b = [b_0 \ \ b_1 \ \ b_2]^T$ is a **block vector** of given data and
# $x = [x_0 \ \ x_1 \ \ x_2]^T$ is a **block vector** of unknowns.

# First, we prepare the objects that will be used to assemble the above algebraic
# structures in DOLFINx. We shall consider a simple problem setup with $A$ representing
# the mass matrix on a finite element space, and $b$ the right-hand side vector that is
# drawn from a uniform distribution.


# +
import pathlib

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from basix.ufl import element
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_matrix_nest,
    assemble_vector_block,
    assemble_vector_nest,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_cube
from fenicsx_pctools.mat import create_splittable_matrix_block
from fenicsx_pctools.utils import vec_to_functions

N = 24
mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
elem = element("Lagrange", mesh.basix_cell(), 1)

# Create product space W = V x V x V
W = [fem.functionspace(mesh, elem) for _ in range(3)]

# Prepare test and trial functions on each subspace
test_functions = [ufl.TestFunction(V) for V in W]
trial_functions = [ufl.TrialFunction(V) for V in W]

# Prepare bilinear form(s) for A
a = [[None for _ in range(3)] for _ in range(3)]

for i, (u_i, v_i) in enumerate(zip(trial_functions, test_functions)):
    a[i][i] = u_i * v_i * ufl.dx

# Prepare linear form(s) for b
f = [fem.Function(V) for V in W]

rng = np.random.default_rng()
for i, f_i in enumerate(f):
    f_i.x.array[:] = rng.uniform(size=f_i.x.array.shape)
    f_i.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

L = [f_i * v_i * ufl.dx for f_i, v_i in zip(f, test_functions)]

# Save RHS plots
outdir = pathlib.Path(__file__).resolve().parent.joinpath("rhs-plots")
for i, f_i in enumerate(f):
    with XDMFFile(MPI.COMM_WORLD, outdir.joinpath(f"f{i}.xdmf"), "w") as handle:
        handle.write_mesh(mesh)
        handle.write_function(f_i)

# Create functions for x
u = [fem.Function(V) for V in W]
# -

# Second, we assemble the algebraic system using two different approaches that are
# available in DOLFINx.


# +
a_dolfinx = fem.form(a)
L_dolfinx = fem.form(L)

# Block assembly
A_block = assemble_matrix_block(a_dolfinx)
A_block.assemble()
A_block.setOption(PETSc.Mat.Option.SPD, True)
b_block = assemble_vector_block(L_dolfinx, a_dolfinx)
b_block.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
x_block = b_block.duplicate()

# Nested assembly
A_nest = assemble_matrix_nest(a_dolfinx)
A_nest.assemble()
A_nest.setOption(PETSc.Mat.Option.SPD, True)
b_nest = assemble_vector_nest(L_dolfinx)
for b_sub in b_nest.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
x_nest = b_nest.duplicate()
# -

# The system $A x = b$ is the algebraic counterpart of the variational problem

# ```{math}
#    (I u, v) = (f, v) \quad \forall v \in W = V \times V \times V,
# ```

# where $(\cdot, \cdot)$ denotes the usual $L^2$ inner product on the computational domain
# and $I$ is the identity operator on the finite element space $W$.

# ## Methodology

# In each of the following sections we solve the algebraic problem using a linear system solver,
# we assign the solution vector $x$ to function $u$ and we verify that the $L^2$-norm of the
# error function $e = u - f$ is a small number. Every time we reset the solution vector to start
# with zero initial guess.


# +
def create_solver(A, prefix=None):
    ksp = PETSc.KSP()
    ksp.create(A.getComm())
    ksp.setOperators(A)
    ksp.setOptionsPrefix(prefix)
    return ksp


def verify_solution(u, f):
    for u_i, f_i in zip(u, f):
        e = u_i.copy()  # ghost values included
        with e.x.petsc_vec.localForm() as e_loc, f_i.x.petsc_vec.localForm() as f_loc:
            e_loc.axpy(-1.0, f_loc)
        error_form = fem.form(e**2 * ufl.dx)
        error = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(error_form) ** 0.5, op=MPI.SUM)
        assert np.isclose(error, 0.0, atol=1e-10)


def reset_solution(x):
    x.zeroEntries()
    if x.getType() == "nest":
        for x_sub in x.getNestSubVecs():
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# -

# Our aim is to configure the solver from a common options database using command line
# arguments recognized by PETSc.


# +
opts = PETSc.Options()
# -

# This will give us the possibility to tweak the solver settings at runtime if needed,
# e.g. by parsing a config file (the parser is not implemented in this demo).

# ## Solutions based on block Jacobi method

# Composable "block" solvers in PETSc can be implemented using the preconditioner of type
# ``PETSc.PC.Type.FIELDSPLIT``. In our first example we will use its "additive" variant, which
# corresponds to the block Jacobi method as it relies on the application of

# ```{math}
#    \begin{align}
#    \begin{bmatrix}
#       \mathrm{ksp}(A_{00}, Ap_{00}) & 0 & 0 \\
#       0 & \mathrm{ksp}(A_{11}, Ap_{11}) & 0 \\
#       0 & 0 & \mathrm{ksp}(A_{22}, Ap_{22})
#    \end{bmatrix}
#    .
#    \end{align}
# ```

# Here, we used the simplified notation $\mathrm{ksp}(A, Ap)$ to mean approximately solving
# a linear system using the ``PETSc.KSP`` object with operator $A$ and preconditioner built
# from matrix $Ap$ (which is the same as $A$ by default).
# We will set up the *inner* ``PETSc.KSP`` objects to use a direct solver.

# ```{note}
# The above notation is used also in the official
# [PETSc User-Guide](https://petsc.org/release/docs/manual/ksp/#solving-block-matrices)
# where it is possible to find more details about block solvers.
# ```

# ### Solution using FEniCSx-pctools

# Let us discuss the solver configuration using our ``A_block`` matrix and together with
# the wrappers offered by FEniCSx-pctools. In order to provide the information that defines
# the blocks to the preconditioner, we must create a *splittable* matrix.


# +
A_splittable = create_splittable_matrix_block(A_block, a)
# -

# The returned matrix is of type ``PETSc.Mat.Type.PYTHON``. As such, it can be connected with
# the UFL form ``a`` that is in turn used to create index sets defining the individual blocks.

# Any fieldsplit preconditioner that is supposed to work with a splittable matrix must be
# wrapped using the dedicated class ``fenicsx_pctools.pc.WrappedPC``.
# Any options of such a wrapped preconditioner are then available using ``"wrapped_"`` prefix.


# +
reset_solution(x_block)
ksp = create_solver(A_splittable, prefix="s1_block_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = ""
opts["pc_type"] = "python"
opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
opts.prefixPush("wrapped_")
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "additive"
opts["pc_fieldsplit_block_size"] = 3
for i in range(3):
    opts[f"pc_fieldsplit_{i}_fields"] = i
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "cholesky"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()  # wrapped_
opts.prefixPop()  # s1_block_

ksp.setFromOptions()
ksp.solve(b_block, x_block)
ksp.destroy()

vec_to_functions(x_block, u)  # updates ghost values
verify_solution(u, f)
# -

# We will show that the above wrappers allow to change the fieldsplit strategy purely from
# the options database, without the need to update the problem setup.
#
# ### Solution using "nest" type
#
# Before changing the fieldsplit strategy, we want to show that the above approach can be
# applied to ``A_nest`` matrix without the need to use FEniCSx-pctools.


# +
reset_solution(x_nest)
ksp = create_solver(A_nest, prefix="s1_nest_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = ""
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "additive"
opts["pc_fieldsplit_block_size"] = 3
for i in range(3):
    opts[f"pc_fieldsplit_{i}_fields"] = i
    opts[f"fieldsplit_{i}_ksp_type"] = "preonly"
    opts[f"fieldsplit_{i}_pc_type"] = "cholesky"
    opts[f"fieldsplit_{i}_pc_factor_mat_solver_type"] = "mumps"
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_nest, x_nest)
ksp.destroy()

vec_to_functions(x_nest, u)  # updates ghost values
verify_solution(u, f)
# -

# ## Solutions based on Schur complement method

# In the next sections, we show what happens if we decide to tweak the solver configuration.
# There will be only few changes in the options database when using the splittable matrix,
# but the code modifications in the other case will be much more extensive.

# In order to apply a solution approach based on Schur complements, we have to rewrite
# the linear operator as a **block $2 \times 2$ matrix**

# ```{math}
#    \begin{align}
#    A &=
#    \begin{bmatrix}
#       A'_{00} & A'_{01} \\
#       A'_{10} & A'_{11}
#    \end{bmatrix}
#    \end{align}
# ```

# with individual blocks recombined e.g. in the following way

# ```{math}
#    \begin{align}
#    A'_{00} &=
#    \begin{bmatrix}
#       A_{00} & A_{02} \\
#       A_{20} & A_{22}
#    \end{bmatrix}
#    , &
#    A'_{01} &=
#    \begin{bmatrix}
#       A_{01} \\
#       A_{21}
#    \end{bmatrix}
#    , &
#    A'_{10} &=
#    \begin{bmatrix}
#       A_{10} & A_{12}
#    \end{bmatrix}
#    , &
#    A'_{11} &= A_{11}.
#    \end{align}
# ```

# The right hand side data can be recombined in the corresponding way, i.e.

# ```{math}
#    \begin{align}
#    b &= \begin{bmatrix} b'_0 \\ b'_1 \end{bmatrix}
#    , &
#    b'_0 &= \begin{bmatrix} b_0 \\ b_2 \end{bmatrix}
#    , &
#    b'_1 &= b_1,
#    \end{align}
# ```

# and the same manipulation can be applied to the solution vector.

# We will solve the exact same problem as above, but this time we will apply the Schur
# complement preconditioner. In particular, we will base the preconditioner on the full
# fatorization of the original system and we will use $A'_{11}$ to build a preconditioner
# for the approximate Schur complement (see
# [PETSc User-Guide](https://petsc.org/release/docs/manual/ksp/#solving-block-matrices)
# for details). Any subsystems resulting from the factorization will be solved using
# the conjugate gradient method in combination with the (point) Jacobi preconditioner
# with a tight relative tolerance.


# +
rtol_cg = 1e-10
# -

# ### Solution using FEniCSx-pctools

# We can change the fieldsplit strategy without the need to update anything in the problem setup.

# +
reset_solution(x_block)
ksp = create_solver(A_splittable, prefix="s2_block_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = ""
opts["pc_type"] = "python"
opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
opts.prefixPush("wrapped_")
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "schur"
opts["pc_fieldsplit_schur_fact_type"] = "full"
opts["pc_fieldsplit_schur_precondition"] = "a11"
opts["pc_fieldsplit_0_fields"] = "0, 2"
opts["pc_fieldsplit_1_fields"] = "1"
for i in range(2):
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = rtol_cg
    opts["pc_type"] = "jacobi"
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()  # wrapped_
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_block, x_block)
ksp.destroy()

vec_to_functions(x_block, u)  # updates ghost values
verify_solution(u, f)
# -

# ### Solution using "nest" type

# Let us see what happens if we update the other solver in a similar fashion.

# +
reset_solution(x_nest)
ksp = create_solver(A_nest, prefix="s2_nest_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = ""
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "schur"
opts["pc_fieldsplit_schur_fact_type"] = "full"
opts["pc_fieldsplit_schur_precondition"] = "a11"
opts["pc_fieldsplit_0_fields"] = "0, 2"
opts["pc_fieldsplit_1_fields"] = "1"
for i in range(2):
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = rtol_cg
    opts["pc_type"] = "jacobi"
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()

ksp.setFromOptions()
# ksp.solve(b_nest, x_nest)  # !!! THIS RAISES AN ERROR !!!
ksp.destroy()

# Remove any unused options to suppress warnings from PETSc
unused_opts = [name for name in opts.getAll() if name.startswith("s2_nest_")]
for name in unused_opts:
    opts.delValue(name)
# -

# It is not possible to define the fields in the above way as ``pc_fieldsplit_0_fields``
# combines nonadjacent blocks and the solver would complain that it *could not find index set*.
# To overcome this issue, we would have to reorder the sytem matrix, e.g. by changing the order
# of subspaces in the mixed space.

# Another workaround is to use ``PETSc.PC.setFieldSplitIS()`` to indicate which
# rows/columns of the matrix belong to a particular block. Moreover, it is necessary
# to convert the system matrix into the ``"aij"`` format. However, this transformation
# may be undesirable in some cases, particularly when the original matrix needs to be
# retained for specific postprocessing operations.


# +
reset_solution(x_nest)
ksp = create_solver(A_nest, prefix="s3_nest_")

ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)
nested_IS = A_nest.getNestISs()
composed_is_row = PETSc.IS(A_nest.getComm()).createGeneral(
    np.concatenate((nested_IS[0][0], nested_IS[0][2]))
)
pc.setFieldSplitIS(
    ["0", composed_is_row],
    ["1", nested_IS[0][1]],
)
A_nest.convert("aij")  # required when combining nonadjacent blocks!
pc.setUp()
for i, sub_ksp in enumerate(pc.getFieldSplitSubKSP()):
    sub_ksp.setType("cg")
    sub_ksp.setTolerances(rtol=rtol_cg)
    sub_ksp.getPC().setType("jacobi")
opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_monitor"] = ""
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_nest, x_nest)
ksp.destroy()

vec_to_functions(x_nest, u)  # updates ghost values
verify_solution(u, f)
# -

# The above code works as expected, but we cannot use the options database
# to set up the index sets.


# +
# Destroy any remaining PETSc objects
x_block.destroy()
x_nest.destroy()
PETSc.garbage_cleanup()
# -

# ## Summary

# - We applied two different solution strategies based on the fieldsplit preconditioning
#   to the same algebraic system of equations.
# - We have shown that transitioning from one solver configuration to the other using
#   FEniCSx-pctools is straightforward as it requires only few updates to PETSc options.
# - We have shown that the same transition is more involved if we use the standard
#   "nest" type, as it requires the changes in the problem setup or undesirable steps
#   like converting the matrix type.

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

# ## Elementary block system of equations

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
# $b = [b_0 \ \ b_1 \ \ b_2]^T$ is a **block vector** of given data and finally
# $x = [x_0 \ \ x_1 \ \ x_2]^T$ is another **block vector** of unknowns.

# First, we prepare the objects that will be used to assemble the above algebraic
# structures in DOLFINx. We shall consider a simple problem setup with $A$ representing
# the mass matrix on a finite element space and $b$ corresponding to a constant function
# over the computational domain.


# +
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
from dolfinx.mesh import create_unit_cube
from fenicsx_pctools.mat import create_splittable_matrix_block
from fenicsx_pctools.utils import vec_to_functions

N = 12
mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
elem = element("Lagrange", mesh.basix_cell(), 1)

# Create product space W = V x V x V
W = [fem.functionspace(mesh, elem) for _ in range(3)]

# Prepare test and trial functions on each subspace
test_functions = [ufl.TestFunction(V) for V in W]
trial_functions = [ufl.TrialFunction(V) for V in W]

# Prepare bilinear form(s) for A
a = [[None for _ in range(3)] for _ in range(3)]

for i, (u_i, v_i) in enumerate(zip(test_functions, trial_functions)):
    a[i][i] = u_i * v_i * ufl.dx

# Prepare linear form(s) for b
f = [fem.Function(V) for V in W]

rhs = [
    lambda x: np.sin(2 * np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]),
    lambda x: (x[0]**2 * (1 - x[0])**2) * (x[1]**2 * (1 - x[1])**2),
    lambda x: np.exp(-x[0]**2 - x[1]**2 - x[2]**2),
]
for f_i, rhs in zip(f, rhs):
    f_i.interpolate(rhs)
    f_i.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

L = [f_i * v_i * ufl.dx for f_i, v_i in zip(f, test_functions)]

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
b_block = assemble_vector_block(L_dolfinx, a_dolfinx)
b_block.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
x_block = A_block.createVecRight()

# Nested assembly
A_nest = assemble_matrix_nest(a_dolfinx)
A_nest.assemble()
b_nest = assemble_vector_nest(L_dolfinx)
for b_sub in b_nest.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
x_nest = A_nest.createVecRight()
# -

# The system $A x = b$ is thus the algebraic counterpart of the problem
# $I u = f$, where $I$ denotes the identity operator on the finite element space
# $W = V \times V \times V$. In each of the following sections we solve the algebraic
# problem using a linear system solver represented by the ``PETSc.KSP`` object, we assign
# the solution vector $x$ to function $u$ and we verify that $u = f$.


# +
def create_solver(A, prefix=None):
    ksp = PETSc.KSP()
    ksp.create(A.getComm())
    ksp.setOperators(A)
    ksp.setOptionsPrefix(prefix)
    return ksp


def verify_solution(u, f):
    for u_i, f_i in zip(u, f):
        with u_i.x.petsc_vec.localForm() as u_i_loc, f_i.x.petsc_vec.localForm() as f_i_loc:
            assert np.allclose(u_i_loc.array_r, f_i_loc.array_r, rtol=1e-6)


# -

# Our aim is to configure the solver from a common options database using command line
# arguments recognized by PETSc.


# +
opts = PETSc.Options()
# -

# This will give us the possibility to tweak the solver settings at runtime if needed,
# e.g. by parsing a config file (not implemented in this demo).

# ## Solution based on block Jacobi method

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

# Let us discuss the solver configuration using our ``A_block`` matrix and a couple of
# wrappers from ``fenicsx_pctools``. In order to provide the information that defines
# the blocks to the preconditioner, we must create the so-called *splittable* matrix.


# +
A_splittable = create_splittable_matrix_block(A_block, a)
# -

# The returned matrix is of type ``PETSc.Mat.Type.PYTHON``. As such, it can be connected with
# the UFL form ``a`` that is in turn used to create index sets defining the blocks.

# Any preconditioner that is supposed to work with a splittable matrix should be configured
# as a *wrapped* ``PETSc.PC`` object using the dedicated class ``fenicsx_pctools.pc.WrappedPC``.
# The same holds for inner preconditioners that operate on extracted submatrices that are
# of the same type. The options of wrapped preconditioners are made accessible using the
# ``"wrapped_"`` prefix.


# +
ksp = create_solver(A_splittable, prefix="s1_block_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = None
opts["pc_type"] = "python"
opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
opts.prefixPush("wrapped_")
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "additive"
for i in range(3):
    opts[f"pc_fieldsplit_{i}_fields"] = i
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()  # wrapped_
opts.prefixPop()  # s1_block_

ksp.setFromOptions()
ksp.solve(b_block, x_block)
ksp.destroy()

vec_to_functions(x_block, u)
verify_solution(u, f)
# -

# ```{note}
# Sometimes it is not necessary to wrap the inner preconditioners, but it is not the case when
# using ``PETSc.PC.Type.LU``.
# ```

# The usage of the "wrapped" preconditioner is necessary to configure the solver at runtime
# from the options database. Next, we show that we cannot achieve this out-of-the-box for
# our ``A_nest`` matrix.


# +
ksp = create_solver(A_nest, prefix="s0_nest_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = None
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "additive"
opts["pc_fieldsplit_block_size"] = 3
for i in range(3):
    opts[f"pc_fieldsplit_{i}_fields"] = i
    opts[f"fieldsplit_{i}_ksp_type"] = "preonly"
    opts[f"fieldsplit_{i}_pc_type"] = "lu"
opts.prefixPop()

ksp.setFromOptions()
# ksp.solve(b_nest, x_nest)  # !!! THIS RAISES AN ERROR !!!
ksp.destroy()

# Remove any unused options to suppress warnings from PETSc
unused_opts = [name for name in opts.getAll() if name.startswith("s0_nest_fieldsplit_")]
for name in unused_opts:
    opts.delValue(name)
# -

# It is not possible to define the fields in the above way since matrices of type
# ``PETSc.Mat.Type.NEST`` are not stored in an interlaced fashion and so the solver
# would complain that it *could not find index set*. To overcome this issue, one has
# to use ``PETSc.PC.setFieldSplitIS()`` to indicate exactly which rows/columns of
# the matrix belong to a particular block.


# +
ksp = create_solver(A_nest, prefix="s1_nest_")

ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
nested_IS = A_nest.getNestISs()
pc.setFieldSplitIS(
    ["0", nested_IS[0][0]],
    ["1", nested_IS[0][1]],
    ["2", nested_IS[0][2]],
)
for i, sub_ksp in enumerate(pc.getFieldSplitSubKSP()):
    sub_ksp.setType("preonly")
    sub_ksp.getPC().setType("lu")
opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_monitor"] = None
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_nest, x_nest)
ksp.destroy()

vec_to_functions(x_nest, u)
verify_solution(u, f)
# -

# Nothing is wrong with the above code except the fact that we cannot use the options
# database to set up the index sets.

# In the next section we show what happens if we decide to tweak the solver configuration.
# There will be only few changes in the options database when using the splittable matrix,
# but the code modifications in the other case will be much more extensive.


# ## Solution based on Schur complement method

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
#       A_{00} & A_{01} \\
#       A_{10} & A_{11}
#    \end{bmatrix}
#    , &
#    A'_{01} &=
#    \begin{bmatrix}
#       A_{02} \\
#       A_{12}
#    \end{bmatrix}
#    , &
#    A'_{10} &=
#    \begin{bmatrix}
#       A_{20} & A_{21}
#    \end{bmatrix}
#    , &
#    A'_{11} &= A_{22}.
#    \end{align}
# ```

# The right hand side data can be recombined in the corresponding way, i.e.

# ```{math}
#    \begin{align}
#    b &= \begin{bmatrix} b'_0 \\ b'_1 \end{bmatrix}
#    , &
#    b'_0 &= \begin{bmatrix} b_0 \\ b_1 \end{bmatrix}
#    , &
#    b'_1 &= b_2,
#    \end{align}
# ```

# and the same manipulation can be applied to the solution vector.

# We will solve the exact same problem as above, but this time we will apply the Schur
# complement preconditioner. In particular, we will base the preconditioner on the full
# fatorization of the original system and we will use $A'_{11}$ to build a preconditioner
# for the approximate Schur complement (see
# [PETSc User-Guide](https://petsc.org/release/docs/manual/ksp/#solving-block-matrices)
# for details). Any subsystems resulting from the factorization will be solved using
# the conjugate gradient method in combination with the (point) Jacobi preconditioner.


# +
ksp = create_solver(A_splittable, prefix="s2_block_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = None
opts["pc_type"] = "python"
opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
opts.prefixPush("wrapped_")
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "schur"
opts["pc_fieldsplit_schur_fact_type"] = "full"
opts["pc_fieldsplit_schur_precondition"] = "a11"
opts["pc_fieldsplit_0_fields"] = "0, 1"
opts["pc_fieldsplit_1_fields"] = "2"
for i in range(2):
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "cg"
    opts["pc_type"] = "jacobi"
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()  # wrapped_
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_block, x_block)
ksp.destroy()

vec_to_functions(x_block, u)
verify_solution(u, f)
# -

# The same strategy can be applied to the system with our ``A_nest`` matrix,
# but we have to manipulate the index sets which makes it cumbersome to use
# if we wish to test diverse solution strategies quickly.

# ```{warning}
# The setup presented below will work as we concatenate the two neighboring index sets.
# If we had decided to combine for example the index sets corresponding to block indices
# 0 and 2, it would have been necessary to convert the system matrix to a different format.
# ```

# +
ksp = create_solver(A_nest, prefix="s2_nest_")

ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("fieldsplit")
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)
nested_IS = A_nest.getNestISs()
composed_is_row = PETSc.IS(A_nest.getComm()).createGeneral(
    np.concatenate((nested_IS[0][0], nested_IS[0][1]))
)
pc.setFieldSplitIS(
    ["0", composed_is_row],
    ["1", nested_IS[0][2]],
)
# A_nest.convert("aij")  # required when combining non-neighboring index sets
pc.setUp()
for i, sub_ksp in enumerate(pc.getFieldSplitSubKSP()):
    sub_ksp.setType("cg")
    sub_ksp.getPC().setType("jacobi")
opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_monitor"] = None
opts.prefixPop()

ksp.setFromOptions()
ksp.solve(b_nest, x_nest)
ksp.destroy()

vec_to_functions(x_nest, u)
verify_solution(u, f)

# Destroy any remaining PETSc objects
PETSc.garbage_cleanup()
# -

# ## Other benefits

# The wrappers discussed above can be used to build advanced custom preconditioners.
# A few of those have been delivered as part of the package, so do not hesitate to explore
# the rest of the demos to find out more.

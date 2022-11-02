# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Motivation

# In this introductory demo we will try to answer the following question for you:

# ```{admonition} Question
# *Why and when should I use FEniCSx PC Tools?*
# ```

# ## Exemplary block system of equations

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
import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_cube
from dolfiny.function import vec_to_functions

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block


N = 2#
mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
elem = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

# Create product space W = V x V x V
W = [fem.FunctionSpace(mesh, elem) for _ in range(3)]

# Prepare test and trial functions on each subspace
test_functions = [ufl.TestFunction(V) for V in W]
trial_functions = [ufl.TrialFunction(V) for V in W]

# Prepare bilinear form(s) for A
a = [[None for _ in range(3)] for _ in range(3)]

for i, (u_i, v_i) in enumerate(zip(test_functions, trial_functions)):
    a[i][i] = u_i * v_i * ufl.dx

# Prepare linear form(s) for b
f = [fem.Function(V) for V in W]

for i, f_i in enumerate(f):
    with f_i.vector.localForm() as f_i_loc:
        f_i_loc.set(i + 1)  # set `f_i = i + 1` including ghost values

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
A_block = fem.petsc.assemble_matrix_block(a_dolfinx)
A_block.assemble()
b_block = fem.petsc.assemble_vector_block(L_dolfinx, a_dolfinx)
x_block = A_block.createVecRight()

# Nested assembly
A_nest = fem.petsc.assemble_matrix_nest(a_dolfinx)
A_nest.assemble()
b_nest = fem.petsc.assemble_vector_nest(L_dolfinx)
for b_sub in b_nest.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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
        with u_i.vector.localForm() as u_i_loc, f_i.vector.localForm() as f_i_loc:
            #print(f"[rank {mesh.comm.rank}] u_{i} = {u_i_loc.array_r}")#FIXME: Remove!
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
#       \mathrm{ksp}(A_{00}, P_{00}) & 0 & 0 \\
#       0 & \mathrm{ksp}(A_{11}, P_{11}) & 0 \\
#       0 & 0 & \mathrm{ksp}(A_{22}, P_{22})
#    \end{bmatrix}
#    .
#    \end{align}
# ```

# Here, we used the simplified notation $\mathrm{ksp}(A, P)$ to mean approximately solving
# a linear system using the ``PETSc.KSP`` object with operator $A$ and preconditioner built
# from matrix $P$ (which is the same as $A$ by default).
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

# The returned matrix is of type ``PETSc.Mat.Type.PYTHON``. As such, it can be endowed with
# the UFL form ``a`` that is in turn used to create index sets defining the blocks.

# Any preconditioner that is supposed to work with a splittable matrix should be configured
# as a *wrapped* ``PETSc.PC`` object using the dedicated class ``fenicsx_pctools.WrappedPC``.
# The same holds for inner preconditioners that operate on extracted submatrices that are
# of the same type. The options of wrapped preconditioners are made accessible using the
# ``"wrapped_"`` prefix.


# +
ksp = create_solver(A_splittable, prefix="s1_block_")

opts.prefixPush(ksp.getOptionsPrefix())
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = None
opts["pc_type"] = "python"
opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
opts.prefixPush("wrapped_")
opts["pc_type"] = "fieldsplit"
opts["pc_fieldsplit_type"] = "additive"
for i in range(3):
    opts[f"pc_fieldsplit_{i}_fields"] = i
    opts.prefixPush(f"fieldsplit_{i}_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "python"
    opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
    opts.prefixPush("wrapped_")
    opts["pc_type"] = "lu"
    opts.prefixPop()  # wrapped_
    opts.prefixPop()  # fieldsplit_{i}_
opts.prefixPop()  # wrapped_
opts.prefixPop()

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

# The excess use of the ``"wrapped_"`` prefix in the above example is the price to pay to
# get things configurable from the options database. Next, we show that we cannot achieve
# this out-of-the-box for our ``A_nest`` matrix.


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

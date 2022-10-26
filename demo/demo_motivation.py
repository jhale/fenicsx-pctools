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

# In this introductory demo we shall try to answer the following question:

# <p style="text-align:center;font"><i>Why and when should you use FEniCSx PC Tools?</i></p>

# ## Construction of a system of equations with block structure

# In what follows, we will solve a system of algebraic equations $A x = b$,
# where the linear operator is given as a $3 \times 3$ **block** matrix

# $$
#    \begin{align}
#    A &=
#    \begin{bmatrix}
#       A_{00} & A_{01} & A_{02} \\
#       A_{10} & A_{11} & A_{12} \\
#       A_{20} & A_{21} & A_{22}
#    \end{bmatrix}
#    \end{align}
# $$

# representing a discretized system of partial differential equations (PDE),
# $b = [b_0, b_1, b_2]^T$ is the block vector of given data, and $x = [x_0, x_1, x_2]^T$ is
# the block vector of unknowns.

# As a first step of this demo, we build the individual components of the above system in DOLFINx.

# +
from mpi4py import MPI

from dolfinx.mesh import create_unit_square


comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 8, 8)
# -

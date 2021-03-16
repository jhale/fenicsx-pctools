#!/usr/bin/env python3
from mpi4py import MPI

import dolfinx
import ufl

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)

V_el = ufl.VectorElement("CG", ufl.triangle, 2)
V = dolfinx.FunctionSpace(mesh, V_el) 

U_el = ufl.FiniteElement("BDM", ufl.triangle, 1)
U = dolfinx.FunctionSpace(mesh, U_el)

v = dolfinx.Function(V)

v_p = ufl.TrialFunction(V)
w_p = ufl.TestFunction(V)

a = inner(

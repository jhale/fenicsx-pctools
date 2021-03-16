#!/usr/bin/env python3
from mpi4py import MPI

import dolfinx
import ufl

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)

V_el = ufl.VectorElement("CG", ufl.triangle, 2)
V = dolfinx.FunctionSpace(mesh, V_el) 

U_el = ufl.FiniteElement("BDM", ufl.triangle, 1)
U = dolfinx.FunctionSpace(mesh, U_el)

v = dolfinx.Function(V)

v_p = ufl.TrialFunction(U)
w_p = ufl.TestFunction(U)

n = ufl.FacetNormal(mesh)

dS = ufl.Measure("dS", mesh)
ds = ufl.Measure("ds", mesh)
a = ufl.inner(ufl.inner(v_p("+"), n("+")), ufl.inner(w_p("+"), n("+")))*dS + ufl.inner(ufl.inner(v_p, n), ufl.inner(w_p, n))*ds
L = ufl.inner(ufl.inner(v("+"), n("+")), ufl.inner(w_p("+"), n("+")))*dS + ufl.inner(ufl.inner(v, n), ufl.inner(w_p, n))*ds

A = dolfinx.fem.assemble_matrix(a)
A.assemble()
print(A.convert('dense').getDenseArray())

b = dolfinx.fem.assemble_vector(L)
print(b.array)

#!/usr/bin/env python3
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem
import ufl
from ufl import inner, dot

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)

V_el = ufl.VectorElement("CG", ufl.triangle, 2)
V = dolfinx.fem.FunctionSpace(mesh, V_el) 

U_el = ufl.FiniteElement("BDM", ufl.triangle, 1)
U = dolfinx.fem.FunctionSpace(mesh, U_el)

def data(x):
    return (x[0], x[1])

v = dolfinx.fem.Function(V)
v.interpolate(data)

v_p = ufl.TrialFunction(U)
w_p = ufl.TestFunction(U)

n = ufl.FacetNormal(mesh)

dS = ufl.Measure("dS", mesh, metadata={"quadrature_degree" : 1})
ds = ufl.Measure("ds", mesh, metadata={"quadrature_degree" : 1})
a = inner(dot(v_p("+"), n("+")), dot(w_p("+"), n("+")))*dS + inner(dot(v_p, n), dot(w_p, n))*ds
L = inner(dot(v("+"), n("+")), dot(w_p("+"), n("+")))*dS + inner(dot(v, n), dot(w_p, n))*ds

A = dolfinx.fem.petsc.assemble_matrix(fem.form(a))
A.assemble()

b = dolfinx.fem.petsc.assemble_vector(fem.form(L))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

PETSc.Options().setValue("ksp_view", "")
PETSc.Options().setValue("ksp_monitor_true_residual", "")
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
# Slow
#solver.setType("preonly")
#solver.getPC().setType("lu")
#solver.getPC().setFactorSolverType("mumps")
# Seems to work well
#solver.setType("richardson")
#solver.getPC().setType("jacobi")
# Also seems to work well.
solver.setType("cg")
solver.getPC().setType("jacobi")
solver.setFromOptions()

v_p_h = dolfinx.fem.Function(U)
solver.solve(b, v_p_h.vector)

#!/usr/bin/env python3

import numpy as np

import ufl
from dolfinx import fem
from dolfinx.cpp.mesh import CellType, GhostMode
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc


def main(N, scheme="CR"):
    mesh = create_unit_square(MPI.COMM_WORLD, N, 2 * N, CellType.triangle, GhostMode.shared_facet)

    if scheme == "CR":
        FEv = ufl.VectorElement("CR", ufl.triangle, 1)
        FEp = ufl.FiniteElement("DG", ufl.triangle, 0)
    else:  # "TH"
        FEv = ufl.VectorElement("P", ufl.triangle, 2)
        FEp = ufl.FiniteElement("P", ufl.triangle, 1)
    Vv = fem.FunctionSpace(mesh, FEv)
    Vp = fem.FunctionSpace(mesh, FEp)
    v = fem.Function(Vv, name="v")
    p = fem.Function(Vp, name="p")

    def bndry_top(x):
        tol = 1.0 / N / 10.0  # to omit dofs in the corners of the domain
        return np.logical_and(np.isclose(x[1], 1.0), x[0] > tol)

    def bndry_bottom(x):
        tol = 1.0 / N / 10.0  # to omit dofs in the corners of the domain
        return np.logical_and(np.isclose(x[1], 0.0), x[0] > tol)

    def bndry_left(x):
        return np.isclose(x[0], 0.0)

    def bndry_right(x):
        return np.isclose(x[0], 1.0)

    bdofsVv_top = fem.locate_dofs_geometrical(Vv, bndry_top)
    bdofsVv_bottom = fem.locate_dofs_geometrical(Vv, bndry_bottom)
    bdofsVv_left = fem.locate_dofs_geometrical(Vv, bndry_left)
    bdofsVv_right = fem.locate_dofs_geometrical(Vv, bndry_right)

    def v_left_eval(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 0.25 * x[1] * (1.0 - x[1])
        return values

    v_zero = fem.Function(Vv, name="v_zero")
    v_left = fem.Function(Vv, name="v_left")
    v_left.interpolate(v_left_eval)

    bcs = [
        fem.DirichletBC(v_left, bdofsVv_left),
        fem.DirichletBC(v_zero, np.hstack((bdofsVv_top, bdofsVv_bottom))),
    ]

    v_te = ufl.TestFunction(Vv)
    p_te = ufl.TestFunction(Vp)

    # Stokes with full stress tensor (physically motivated form, CR scheme needs stabilization)
    I = ufl.Identity(2)
    D = lambda v: ufl.sym(ufl.grad(v))
    T = lambda p, v: -p * I + 2.0 * D(v)
    F0 = ufl.inner(T(p, v), D(v_te)) * ufl.dx

    # Stabilization term required by CR scheme (its contribution is zero for TH scheme)
    h = ufl.FacetArea(mesh)
    F0 += (1.0 / h) * ufl.inner(ufl.jump(v), ufl.jump(v_te)) * ufl.dS

    # Stokes with Laplace term (classical form, CR scheme works fine)
    # F0 = (ufl.inner(ufl.grad(v), ufl.grad(v_te)) - p * ufl.div(v_te)) * ufl.dx

    F1 = -ufl.div(v) * p_te * ufl.dx
    F_form = [F0, F1]

    v_tr = ufl.TrialFunction(Vv)
    p_tr = ufl.TrialFunction(Vp)
    J_form = [
        [ufl.derivative(F0, v, v_tr), ufl.derivative(F0, p, p_tr)],
        [ufl.derivative(F1, v, v_tr), None]
    ]

    def snesctx_vec_to_functions(x, u):
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] = x.array_r[offset : offset + size_local]
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

    def snesctx_F(snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        snesctx_vec_to_functions(x, [v, p])
        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.assemble_vector_block(F, F_form, J_form, bcs, x0=x, scale=-1.0)

    def snesctx_J(snes, x, J, P):
        J.zeroEntries()
        fem.assemble_matrix_block(J, J_form, bcs, diagonal=1.0)
        J.assemble()

    Jmat = fem.create_matrix_block(J_form)
    Fvec = fem.create_vector_block(F_form)
    x0 = fem.create_vector_block(F_form)

    opts = PETSc.Options()
    opts["snes_monitor"] = None
    opts["snes_converged_reason"] = None
    opts["ksp_monitor"] = None
    opts["ksp_converged_reason"] = None
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts["mat_mumps_icntl_4"] = 0  # verbosity
    opts["mat_mumps_icntl_14"] = 50  # % increase in the estimated working space (default: 20)
    solver = PETSc.SNES().create(mesh.mpi_comm())
    solver.setFunction(snesctx_F, Fvec)
    solver.setJacobian(snesctx_J, J=Jmat, P=None)
    solver.setFromOptions()

    solver.solve(None, x0)

    # snesctx_vec_to_functions(x0, [v, p])
    # with XDMFFile(mesh.mpi_comm(), "v.xdmf", "w") as xfile:
    #     mesh.topology.create_connectivity_all()
    #     xfile.write_mesh(mesh)
    #     xfile.write_function(v)

    return 0 if solver.getConvergedReason() > 0 else 1


if __name__ == "__main__":
    num_total = 0
    num_failed = 0

    for N in range(4, 21):
        print(f"N = {N}")
        num_failed += main(N, scheme="CR")  # CR ... Crouzeix-Raviart, TH ... Taylor-Hood
        num_total += 1

    print(f"Number of failed jobs: {num_failed} (out of {num_total})")

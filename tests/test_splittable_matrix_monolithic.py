# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc

import pytest

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem
from dolfinx.fem.function import Function, functionspace
from dolfinx.fem.petsc import create_matrix
from dolfinx.mesh import create_unit_square
from fenicsx_pctools.mat.splittable import create_splittable_matrix_monolithic


def _create_constant(function_space, value):
    f = Function(function_space)
    with f.x.petsc_vec.localForm() as f_local:
        f_local.set(value)
    return f


@pytest.fixture
def get_vector_space():
    def _get_vector_space(mesh, equal_discretization):
        CG1 = element("Lagrange", mesh.basix_cell(), 1)
        CG2 = element("Lagrange", mesh.basix_cell(), 2)

        if equal_discretization:
            ME = element("Lagrange", mesh.basix_cell(), 1, shape=(3,))
        else:
            components = (CG1, CG1, CG2)
            ME = mixed_element(components)

        return functionspace(mesh, ME)

    return _get_vector_space


@pytest.mark.parametrize(
    "equal_discretization",
    [True, pytest.param(False, marks=pytest.mark.xfail(raises=NotImplementedError))],
    ids=["equal subspaces", "different subspaces"],
)
def test_nested_fieldsplit(get_vector_space, equal_discretization, comm):
    mesh = create_unit_square(comm, 4, 4)
    V = get_vector_space(mesh, equal_discretization)
    bs = V.dofmap.dof_layout.block_size
    if equal_discretization:
        assert bs == 3
    else:
        assert bs == 1

    v_target = Function(V)
    v0, v1, v2 = v_target.split()
    v0.interpolate(_create_constant(v0.function_space.collapse()[0], 1.0))
    v1.interpolate(_create_constant(v1.function_space.collapse()[0], 2.0))
    v2.interpolate(_create_constant(v2.function_space.collapse()[0], 3.0))

    v_tr, v_te = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(v_tr, v_te) * ufl.dx
    L = ufl.inner(v_target, v_te) * ufl.dx

    a_dolfinx = fem.form(a)
    A = create_matrix(a_dolfinx)
    fem.petsc.assemble_matrix(A, a_dolfinx)
    A.assemble()

    A_splittable = create_splittable_matrix_monolithic(A, a)

    L_dolfinx = fem.form(L)
    b = fem.petsc.assemble_vector(L_dolfinx)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A_splittable)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitFields(bs, ["0", [0, 2]], ["1", 1])

    pc.setUp()

    ksp_sub = pc.getFieldSplitSubKSP()
    ksp_sub[0].setType("cg")
    ksp_sub[1].setType("cg")
    pc_sub = [ksp_i.getPC() for ksp_i in ksp_sub]
    pc_sub[1].setType("none")
    pc_sub[0].setType("fieldsplit")
    pc_sub[0].setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    pc_sub[0].setFieldSplitFields(2, ["0", 1], ["1", 0])  # ! Changed block size & field indices !
    for i, ksp_i in enumerate(pc_sub[0].getFieldSplitSubKSP()):
        assert ksp_i.prefix == f"fieldsplit_0_fieldsplit_{i}_"
        ksp_i.setType("cg")
        ksp_i.getPC().setType("none")

    x = b.copy()
    x.zeroEntries()
    ksp.solve(b, x)

    v_target.x.petsc_vec.axpy(-1.0, x)
    assert v_target.x.petsc_vec.norm() == pytest.approx(0.0, abs=1.0e-10)

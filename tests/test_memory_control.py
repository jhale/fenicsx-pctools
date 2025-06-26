# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix_block
from dolfinx.mesh import create_unit_square
from fenicsx_pctools.mat.splittable import create_splittable_matrix_block


@pytest.fixture
def a(comm):
    mesh = create_unit_square(comm, 2, 2)
    CG1 = fem.functionspace(mesh, ("Lagrange", 1))
    V = [CG1, CG1]

    trial_functions = tuple(map(lambda V_sub: ufl.TrialFunction(V_sub), V))
    test_functions = tuple(map(lambda V_sub: ufl.TestFunction(V_sub), V))

    a = np.full((2, 2), None).tolist()
    for i, (v_tr, v_te) in enumerate(zip(trial_functions, test_functions)):
        a[i][i] = ufl.inner(v_tr, v_te) * ufl.dx

    return a


def test_destroy_mat(a):
    A = assemble_matrix_block(fem.form(a))
    A.assemble()
    A_splittable = create_splittable_matrix_block(A, a)
    assert A_splittable.refcount == 1
    assert A.refcount == 1

    ksp = PETSc.KSP().create(A_splittable.comm)
    ksp.setOperators(A_splittable, A_splittable)  # adds 2 new references
    assert A_splittable.refcount == 3
    assert A.refcount == 1

    ksp.destroy()  # removes the 2 references added above
    assert A_splittable.refcount == 1
    assert A.refcount == 1

    # Make duplicates before removing the originals
    B_splittable = A_splittable.duplicate()
    B = B_splittable.getPythonContext().Mat

    # Remove the original wrapper
    A_splittable.destroy()
    assert A_splittable.refcount == 0
    assert A.refcount == 1

    # Only now remove the wrapped matrix
    A.destroy()
    assert A.refcount == 0

    # Check that duplicates were left untouched
    assert B_splittable.refcount == 1
    assert B.refcount == 1

    # Artificially increase the number of references for the wrapper
    B_splittable.incRef()
    assert B_splittable.refcount == 2
    assert B.refcount == 1

    # Try to remove the duplicates
    B_splittable.destroy()  # NOT called as the reference count is >1 !?
    assert B_splittable.refcount == 0
    assert B.refcount == 1
    B.destroy()
    assert B.refcount == 0


def test_destroy_ises(a):
    A = assemble_matrix_block(fem.form(a))
    A.assemble()
    A_splittable = create_splittable_matrix_block(A, a)
    assert A_splittable.refcount == 1
    assert A.refcount == 1

    isrows, iscols = A_splittable.getPythonContext().ISes
    for iset_row, iset_col in zip(isrows, iscols):
        assert iset_row.refcount == 1
        assert iset_col.refcount == 1

    A_splittable.destroy()
    assert A_splittable.refcount == 0
    assert A.refcount == 1
    for iset_row, iset_col in zip(isrows, iscols):
        assert iset_row.refcount == 0
        assert iset_col.refcount == 0

    A.destroy()
    assert A.refcount == 0

import numpy as np
import pytest
import ufl

from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_square

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block


@pytest.fixture
def a(comm):
    mesh = create_unit_square(comm, 2, 2)
    CG1 = fem.FunctionSpace(mesh, ufl.FiniteElement("CG", mesh.ufl_cell(), 1))
    V = [CG1, CG1]

    trial_functions = tuple(map(lambda V_sub: ufl.TrialFunction(V_sub), V))
    test_functions = tuple(map(lambda V_sub: ufl.TestFunction(V_sub), V))

    a = np.full((2, 2), None).tolist()
    for i, (v_tr, v_te) in enumerate(zip(trial_functions, test_functions)):
        a[i][i] = ufl.inner(v_tr, v_te) * ufl.dx

    return a


def test_refcount(a):
    A = fem.petsc.assemble_matrix_block(fem.form(a))
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

    # Make duplicates before removing the original matrices
    B_splittable = A_splittable.duplicate()
    B = B_splittable.getPythonContext().Mat

    # Try to remove the originals
    A_splittable.destroy()  # works as expected, both objects get destroyed
    assert A_splittable.refcount == 0
    assert A.refcount == 0

    # Duplicates left untouched
    assert B_splittable.refcount == 1
    assert B.refcount == 1

    # Artificially increase the number of references for B_splittable
    B_splittable.incRef()
    assert B_splittable.refcount == 2
    assert B.refcount == 1

    # Try to remove the duplicates when the reference counts are not equal
    B_splittable.destroy()  # !!! DOESN'T WORK AS EXPECTED !!!
    assert B_splittable.refcount == 0
    with pytest.raises(AssertionError):
        assert B.refcount == 0  # B.refcount == 1


def test_ises(a):
    A = fem.petsc.assemble_matrix_block(fem.form(a))
    A.assemble()

    A_splittable = create_splittable_matrix_block(A, a)
    assert A_splittable.refcount == 1
    assert A.refcount == 1

    isrows, iscols = A_splittable.getPythonContext().ISes
    for iset_r, iset_c in zip(isrows, iscols):
        assert iset_r.refcount == 1
        assert iset_c.refcount == 1

    A_splittable.destroy()
    assert A_splittable.refcount == 0
    assert A.refcount == 0
    for iset_r, iset_c in zip(isrows, iscols):
        assert iset_r.refcount == 0
        assert iset_c.refcount == 0

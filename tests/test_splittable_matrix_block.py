import numpy as np
import pytest

import ufl
from dolfinx import cpp, fem
from dolfinx.fem.function import Function, FunctionSpace
from dolfinx.mesh import create_unit_square
from fenicsx_pctools.mat.splittable import create_splittable_matrix_block

from petsc4py import PETSc


@pytest.fixture
def get_block_space():
    def _get_block_space(mesh, equal_discretization):
        CG1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        CG2 = ufl.FiniteElement("CG", mesh.ufl_cell(), 2)

        components = (CG1, CG1, CG1) if equal_discretization else (CG1, CG1, CG2)

        return tuple([FunctionSpace(mesh, FE) for FE in components])

    return _get_block_space


@pytest.mark.parametrize(
    "equal_discretization", [True, False], ids=["equal subspaces", "different subspaces"]
)
def test_nested_fieldsplit(get_block_space, equal_discretization, comm):
    mesh = create_unit_square(comm, 4, 4)

    V = get_block_space(mesh, equal_discretization)

    v_target = tuple([Function(Vsub) for Vsub in V])
    for i, vsub in enumerate(v_target):
        with vsub.vector.localForm() as vsub_local:
            vsub_local.set(i)

    a = np.full((3, 3), None).tolist()
    L = np.full(3, None).tolist()
    for i, (Vsub, vsub) in enumerate(zip(V, v_target)):
        v_tr, v_te = ufl.TrialFunction(Vsub), ufl.TestFunction(Vsub)
        a[i][i] = ufl.inner(v_tr, v_te) * ufl.dx
        L[i] = ufl.inner(vsub, v_te) * ufl.dx

    A = create_splittable_matrix_block(a)
    A.assemble()

    L_dolfinx = [fem.form(L) for L in L]
    imaps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L_dolfinx
    ]
    b = fem.petsc.create_vector_block(L_dolfinx)
    b.set(0.0)
    b_local = cpp.la.petsc.get_local_vectors(b, imaps)
    for b_sub, L_sub in zip(b_local, L_dolfinx):
        fem.assemble_vector(b_sub, L_sub)
    cpp.la.petsc.scatter_local_vectors(b, b_local, imaps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp = PETSc.KSP()
    ksp.create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    A_ctx = A.getPythonContext()
    composed_is_row = PETSc.IS(comm).createGeneral(
        np.concatenate((A_ctx.ISes[0][0].indices, A_ctx.ISes[0][2].indices))
    )
    composed_is_col = PETSc.IS(comm).createGeneral(
        np.concatenate((A_ctx.ISes[1][0].indices, A_ctx.ISes[1][2].indices))
    )
    assert np.all(composed_is_row.indices == composed_is_col.indices)
    pc.setFieldSplitIS(
        ["0", composed_is_row],
        ["1", A_ctx.ISes[0][1]],
    )
    pc.setUp()

    ksp_sub = pc.getFieldSplitSubKSP()
    ksp_sub[0].setType("preonly")
    ksp_sub[1].setType("cg")
    pc_sub = [ksp_i.getPC() for ksp_i in ksp_sub]
    pc_sub[1].setType("jacobi")
    pc_sub[0].setType("fieldsplit")
    pc_sub[0].setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    Asub = pc_sub[0].getOperators()[0]
    Asub_ctx = Asub.getPythonContext()
    pc_sub[0].setFieldSplitIS(
        ["0", Asub_ctx.ISes[0][1]],
        ["1", Asub_ctx.ISes[0][0]],
    )
    for i, ksp_i in enumerate(pc_sub[0].getFieldSplitSubKSP()):
        assert ksp_i.prefix == f"fieldsplit_0_fieldsplit_{i}_"
        ksp_i.setType("cg")
        ksp_i.getPC().setType("jacobi")

    x = b.copy()
    x.zeroEntries()
    ksp.solve(b, x)

    imaps = [(V_sub.dofmap.index_map, V_sub.dofmap.index_map_bs) for V_sub in V]
    target_vec = cpp.fem.petsc.create_vector_block(imaps)
    target_vec.set(0.0)
    cpp.la.petsc.scatter_local_vectors(
        target_vec, list(map(lambda f_sub: f_sub.vector.array, v_target)), imaps
    )

    target_vec.axpy(-1.0, x)
    assert target_vec.norm() == pytest.approx(0.0, abs=1.0e-08)

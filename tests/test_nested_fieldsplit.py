import itertools

import numpy as np
import pytest

import ufl
from dolfinx import cpp, fem
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_block, assemble_matrix_nest
from dolfinx.mesh import CellType, create_unit_square
from fenicsx_pctools.mat.splittable import (
    create_splittable_matrix_block,
    create_splittable_matrix_monolithic,
)

from petsc4py import PETSc


@pytest.fixture(
    params=itertools.product(
        # ["block", "nest", "monolithic"],
        ["block", "nest"],
        [CellType.triangle, CellType.quadrilateral],
    )
)
def space(request, comm):
    if comm.rank == 0:
        print(request.param)
    structure, cell_type = request.param

    class MixedSpace(object):
        def __init__(self, mesh, structure):
            self.mesh = mesh
            self.structure = structure

            family = "P" if cell_type == cpp.mesh.CellType.triangle else "Q"
            CG1 = ufl.FiniteElement(family, mesh.ufl_cell(), 1)
            CG2 = ufl.FiniteElement(family, mesh.ufl_cell(), 2)
            FE0 = ufl.MixedElement([CG2, CG2])
            FE1 = CG1
            FE2 = CG2
            FE3 = ufl.TensorElement(
                family, mesh.ufl_cell(), 1, shape=(2, 2), symmetry={(1, 0): (0, 1)}
            )

            if structure == "monolithic":
                self._V = FunctionSpace(mesh, ufl.MixedElement([FE0, FE1, FE2, FE3]))
            elif structure in ["block", "nest"]:
                self._V = (
                    FunctionSpace(mesh, FE0),
                    FunctionSpace(mesh, FE1),
                    FunctionSpace(mesh, FE2),
                    FunctionSpace(mesh, FE3),
                )

        def __call__(self):
            return self._V

        @staticmethod
        def create_constant(function_space, value):
            f = Function(function_space)
            with f.vector.localForm() as f_local:
                f_local.set(value)
            return f

    ghost_mode = cpp.mesh.GhostMode.shared_facet
    mesh = create_unit_square(comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode)
    return MixedSpace(mesh, structure)


@pytest.fixture
def A(space):
    V = space()
    if space.structure == "monolithic":
        v_tr, v_te = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(v_tr, v_te) * ufl.dx
    else:
        trial_functions = tuple(map(lambda V_sub: ufl.TrialFunction(V_sub), V))
        test_functions = tuple(map(lambda V_sub: ufl.TestFunction(V_sub), V))
        a = np.full((4, 4), None).tolist()
        for i, (v_tr, v_te) in enumerate(zip(trial_functions, test_functions)):
            a[i][i] = ufl.inner(v_tr, v_te) * ufl.dx

    a_dolfinx = fem.form(a)

    A = {
        "block": assemble_matrix_block,
        "monolithic": assemble_matrix,
        "nest": assemble_matrix_nest,
    }[space.structure](a_dolfinx)
    A.assemble()

    A_splittable = {
        "block": create_splittable_matrix_block,
        "monolithic": create_splittable_matrix_monolithic,
        "nest": lambda A, a_dolfinx: A,
    }[space.structure](A, a_dolfinx)

    return A_splittable


@pytest.fixture
def target(space):
    V = space()
    if space.structure == "monolithic":
        f = Function(V)
        v0, v1, v2, v3 = f.split()
        V = (
            v0.function_space,
            v1.function_space.collapse(),
            v2.function_space.collapse(),
            v3.function_space,
        )
    else:
        f = tuple(map(lambda V_sub: Function(V_sub), V))
        v0, v1, v2, v3 = f

    v0.sub(0).interpolate(space.create_constant(V[0].sub(0).collapse()[0], 1.0))
    v0.sub(1).interpolate(space.create_constant(V[0].sub(1).collapse()[0], 2.0))
    v1.interpolate(space.create_constant(V[1], 3.0))
    v2.interpolate(space.create_constant(V[2], 4.0))
    v3.sub(0).interpolate(space.create_constant(V[3].sub(0).collapse()[0], 5.0))
    v3.sub(1).interpolate(space.create_constant(V[3].sub(1).collapse()[0], 6.0))
    v3.sub(2).interpolate(space.create_constant(V[3].sub(2).collapse()[0], 7.0))

    return f


@pytest.fixture
def b(space, target):
    V = space()
    if space.structure == "monolithic":
        v_te = ufl.TestFunction(V)

        L = fem.form(ufl.inner(target, v_te) * ufl.dx)

        b = fem.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    else:
        test_functions = tuple(map(lambda V_sub: ufl.TestFunction(V_sub), V))

        L = np.full(4, None).tolist()

        for i, (f_sub, v_te) in enumerate(zip(target, test_functions)):
            L[i] = ufl.inner(f_sub, v_te) * ufl.dx

        L = fem.form(L)

        if space.structure == "block":
            imaps = [
                (
                    form.function_spaces[0].dofmap.index_map,
                    form.function_spaces[0].dofmap.index_map_bs,
                )
                for form in L
            ]
            b = cpp.fem.petsc.create_vector_block(imaps)
            b.set(0.0)
            b_local = cpp.la.petsc.get_local_vectors(b, imaps)
            for b_sub, L_sub in zip(b_local, L):
                fem.assemble_vector(b_sub, L_sub)
            cpp.la.petsc.scatter_local_vectors(b, b_local, imaps)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        elif space.structure == "nest":
            b = fem.petsc.assemble_vector_nest(L)
            for b_sub in b.getNestSubVecs():
                b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        else:
            raise RuntimeError

    return b


# NOTE:
#   For the discussion around recursive field splits with Matnest refer to:
#   - https://petsc-users.mcs.anl.narkive.com/daEIQkM8/recursive-fieldsplit-pcs
#   - https://petsc-users.mcs.anl.narkive.com/OYkmc42S/recursive-field-split-with-matnest
#
#   Matt Knepley about the problem with pure IS solution:
#
#     "The obvious solution is to just combine several ISes to create the field for
#     PCFIELDSPLIT. However, once this is done, they lose their identity as
#     separate fields. Thus, it is not possible to untangle for the second level
#     of FieldSplit that you want. The DM version maintains
#     the field identity at the next level so that we can split hierarchically.
#     So, for the above to work, I think you must use a DM."


@pytest.mark.parametrize(
    "variant",
    (
        "LU",
        "FS 0_1_2_3",  # 1 LEVEL (4 blocks)
        "FS 0-3_1-2",  # 1 LEVEL (2 blocks)
        "FS 0-2_1_3",  # 1 LEVEL (3 blocks)
        "FS 0-2-3_1 1_0-2",  # 2 LEVELS (2 blocks + 2 blocks)
        "FS 0-2-3_1 1_0-2 1_0",  # 3 LEVELS (2 blocks + 2 blocks + 2 blocks)
    ),
)
def test_nested_fieldsplit(space, A, b, target, variant):
    comm = space.mesh.comm

    ksp = PETSc.KSP()
    ksp.create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    opts = PETSc.Options()

    if variant == "LU":

        if space.structure == "nest":
            pytest.skip("Direct solver cannot be used with 'nest' structures")
        else:
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts["wrapped_pc_type"] = "lu"

    elif variant == "FS 0_1_2_3":

        if space.structure == "block":
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = 0
            opts["pc_fieldsplit_1_fields"] = 1
            opts["pc_fieldsplit_2_fields"] = 2
            opts["pc_fieldsplit_3_fields"] = 3
            for i in range(4):
                opts[f"fieldsplit_{i}_ksp_type"] = "cg"
                opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            opts.prefixPop()
        elif space.structure == "nest":
            # opts["pc_type"] = "fieldsplit"
            # opts["pc_fieldsplit_type"] = "additive"
            # opts["pc_fieldsplit_block_size"] = 4
            # opts["pc_fieldsplit_0_fields"] = 0
            # opts["pc_fieldsplit_1_fields"] = 1
            # opts["pc_fieldsplit_2_fields"] = 2
            # opts["pc_fieldsplit_3_fields"] = 3
            # for i in range(4):
            #     opts[f"fieldsplit_{i}_ksp_type"] = "cg"
            #     opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            #
            # NOTE:
            #   Definition of fields in the above way is not possible since MATNEST matrices
            #   are not stored in an interlaced fashion (see p. 76 in the PETSc manual).
            #   Hence, this will raise PETSc error (75) with the complementary message
            #   "Could not find index set". Instead, PCFieldSplitSetIS() has to be used to
            #   indicate exactly which rows/columns of the matrix belong to a particular block.
            pc = ksp.getPC()
            pc.setType("fieldsplit")
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            nested_IS = A.getNestISs()
            pc.setFieldSplitIS(
                ["0", nested_IS[0][0]],
                ["1", nested_IS[0][1]],
                ["2", nested_IS[0][2]],
                ["3", nested_IS[0][3]],
            )
            for i, sub_ksp in enumerate(pc.getFieldSplitSubKSP()):
                assert sub_ksp.prefix == f"fieldsplit_{i}_"
                sub_ksp.setType("cg")
                sub_ksp.getPC().setType("jacobi")
        else:
            pytest.skip(f"Variant '{variant}' needs to be implemented for '{space.structure}'")

    elif variant == "FS 0-3_1-2":

        if space.structure == "block":
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "0, 3"
            opts["pc_fieldsplit_1_fields"] = "1, 2"
            for i in range(2):
                opts[f"fieldsplit_{i}_ksp_type"] = "cg"
                opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            opts.prefixPop()
        else:
            pytest.skip(f"Variant '{variant}' needs to be implemented for '{space.structure}'")

    elif variant == "FS 0-2_1_3":

        if space.structure == "block":
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "0, 2"
            opts["pc_fieldsplit_1_fields"] = "1"
            opts["pc_fieldsplit_2_fields"] = "3"
            for i in range(3):
                opts[f"fieldsplit_{i}_ksp_type"] = "cg"
                opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            opts.prefixPop()
        elif space.structure == "nest":
            if comm.size > 1:
                pytest.skip("MATNEST matrix cannot be coverted in parallel?")
            pc = ksp.getPC()
            pc.setType("fieldsplit")
            pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
            nested_IS = A.getNestISs()
            A.convert("aij")  # this is harsh (and not working in parallel)
            composed_is_row = PETSc.IS(comm).createGeneral(
                np.concatenate((nested_IS[0][0], nested_IS[0][2]))
            )
            pc.setFieldSplitIS(
                ["0", composed_is_row],  # raises "Could not find index set" (with no conversion)
                ["1", nested_IS[0][1]],
                ["2", nested_IS[0][3]],
            )
            for i, sub_ksp in enumerate(pc.getFieldSplitSubKSP()):
                assert sub_ksp.prefix == f"fieldsplit_{i}_"
                sub_ksp.setType("cg")
                sub_ksp.getPC().setType("jacobi")
        else:
            pytest.skip(f"Variant '{variant}' needs to be implemented for '{space.structure}'")

    elif variant == "FS 0-2-3_1 1_0-2":

        if space.structure == "block":
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "0, 2, 3"
            opts["pc_fieldsplit_1_fields"] = "1"
            opts["fieldsplit_0_ksp_type"] = "preonly"
            opts["fieldsplit_0_pc_type"] = "python"
            opts["fieldsplit_0_pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("fieldsplit_0_wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "1"
            opts["pc_fieldsplit_1_fields"] = "0, 2"
            for i in range(2):
                opts[f"fieldsplit_{i}_ksp_type"] = "cg"
                opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            opts.prefixPop()
            opts["fieldsplit_1_ksp_type"] = "cg"
            opts["fieldsplit_1_pc_type"] = "jacobi"
            opts.prefixPop()
        else:
            pytest.skip(f"Variant '{variant}' needs to be implemented for '{space.structure}'")

    elif variant == "FS 0-2-3_1 1_0-2 1_0":

        if space.structure == "block":
            opts["pc_type"] = "python"
            opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "0, 2, 3"
            opts["pc_fieldsplit_1_fields"] = "1"
            opts["fieldsplit_0_ksp_type"] = "preonly"
            opts["fieldsplit_0_pc_type"] = "python"
            opts["fieldsplit_0_pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("fieldsplit_0_wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "additive"
            opts["pc_fieldsplit_0_fields"] = "1"
            opts["pc_fieldsplit_1_fields"] = "0, 2"
            opts["fieldsplit_0_ksp_type"] = "cg"
            opts["fieldsplit_0_pc_type"] = "jacobi"
            opts["fieldsplit_1_ksp_type"] = "preonly"
            opts["fieldsplit_1_pc_type"] = "python"
            opts["fieldsplit_1_pc_python_type"] = "fenicsx_pctools.WrappedPC"
            opts.prefixPush("fieldsplit_1_wrapped_")
            opts["pc_type"] = "fieldsplit"
            opts["pc_fieldsplit_type"] = "schur"
            opts["pc_fieldsplit_0_fields"] = "1"
            opts["pc_fieldsplit_1_fields"] = "0"
            for i in range(2):
                opts[f"fieldsplit_{i}_ksp_type"] = "cg"
                opts[f"fieldsplit_{i}_pc_type"] = "jacobi"
            opts.prefixPop()
            opts.prefixPop()
            opts["fieldsplit_1_ksp_type"] = "cg"
            opts["fieldsplit_1_pc_type"] = "jacobi"
            opts.prefixPop()
        else:
            pytest.skip(f"Variant '{variant}' needs to be implemented for '{space.structure}'")

    else:
        raise NotImplementedError(f"Unknown variant '{variant}'")

    ksp.setFromOptions()

    x = b.copy()
    ksp.solve(b, x)

    if space.structure == "monolithic":
        target_vec = target.vector
    else:
        V = space()
        imaps = [(V_sub.dofmap.index_map, V_sub.dofmap.index_map_bs) for V_sub in V]
        target_vec = cpp.fem.petsc.create_vector_block(imaps)
        target_vec.set(0.0)
        cpp.la.petsc.scatter_local_vectors(
            target_vec, list(map(lambda f_sub: f_sub.vector.array, target)), imaps
        )

    target_vec.axpy(-1.0, x)
    assert target_vec.norm() == pytest.approx(0.0, 1.0e-10)

    # Clean up options database
    for opt in opts.getAll().keys():
        opts.delValue(opt)
    # NOTE: None of the following "destructors" works:
    #   - opts.clear()
    #   - opts.destroy()
    #   - del opts
    # TODO: Consider implementation of an `OptionsManager`, see
    #       https://github.com/firedrakeproject/firedrake/blob/master/firedrake/petsc.py

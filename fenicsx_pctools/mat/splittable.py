# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import abc
import enum
import typing

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import cpp
from dolfinx.fem.function import FunctionSpace


def _extract_spaces(ufl_form: ufl.Form) -> tuple[FunctionSpace, FunctionSpace]:
    """Extract test and trial function spaces from given bilinear form.

    Parameters:
        ufl_form: bilinear form

    Returns:
        test space, trial space
    """
    test_space, trial_space = map(lambda arg: arg.ufl_function_space(), ufl_form.arguments())
    return test_space, trial_space


def _copy_index_sets(isets) -> tuple[list[PETSc.IS], list[PETSc.IS]]:
    """Duplicate items contained in given tuple of lists with index sets for block rows/columns.

    Returns:
        duplicated row index sets and column index sets respectively
    """

    if isets is None:
        return None

    isets_copy = (
        [iset.duplicate() for iset in isets[0]],
        [iset.duplicate() for iset in isets[1]],
    )
    return isets_copy


class _MatrixLayout(enum.IntEnum):
    """Type of the layout of matrix entries."""

    MIX = 1  #: for mixed spaces with more than one subspace and dofmap's block size set to 1
    STRIDE = 2  #: for mixed spaces with dofmap's block size set to the number of subspaces
    BLOCK = 3  #: special layout type reserved for block matrices


def _analyse_block_structure(space: FunctionSpace) -> tuple[int, _MatrixLayout]:
    """Deduce the number of blocks from the number of subspaces of the given space (return 1 if
    the space have no subspaces) and determine :class:`_MatrixLayout` by comparing the number of
    blocks with the dofmap's block size.

    Parameters:
        space: function space to be analysed

    Returns:
        number of blocks (subspaces) and matrix layout type
    """
    num_blocks = space.element.num_sub_elements or 1  # no/zero subspaces -> single block
    bs = space.dofmap.index_map_bs  # 1 for "truly" mixed function spaces

    if bs < num_blocks:
        dof_layout = _MatrixLayout.MIX
    else:
        assert bs == num_blocks
        dof_layout = _MatrixLayout.STRIDE

    return num_blocks, dof_layout


def _find_block_indices(iset: PETSc.IS, isets: tuple[PETSc.IS]) -> tuple[int]:
    """Verify that given *iset* can be obtained as a subset of *isets* and return corresponding
    block indices (i.e. positions in *isets*).

    Parameters:
        iset: index set to be found in *isets*
        isets: iterable containing PETSc IS objects

    Returns:
        positions in *isets* marking index sets that need to be concatenated to obtain *iset*

    Raises:
        ValueError: when *iset* cannot be obtained as a subset of *isets*
    """
    block_ids = []
    comm = iset.comm.tompi4py()
    parent_indices = iset.indices
    search_space = {k: v for k, v in enumerate(isets)}

    # NOTE:
    #   In the following loop, we repeatedly iterate over the search space taking one index set
    #   after the other (child IS), while trying to find it at the beginning of the parent IS.
    #   If there is a match between the two, the respective indices are removed from the parent IS,
    #   the child IS is removed from the search space and the corresponding block ID is remembered.
    #   No indices should remain in the parent IS at the end of this process.
    while True:
        found = False
        for i, child in list(search_space.items()):
            child_indices = child.indices
            child_size = child_indices.shape[0]
            parent_size = parent_indices.shape[0]
            local_match = child_size <= parent_size and np.array_equal(
                parent_indices[:child_size], child_indices
            )
            if comm.allreduce(local_match, op=MPI.LAND):
                found = True
                parent_indices = parent_indices[child_size:]  # remove indices from parent IS
                search_space.pop(i)  # remove child IS from search space
                block_ids.append(i)  # remember the position
        if not found:
            break

    if comm.allreduce(len(parent_indices), op=MPI.SUM) > 0:
        raise ValueError(f"Unable to identify {iset} as a subset of {isets}")

    return tuple(block_ids)


class SplittableMatrixBase(metaclass=abc.ABCMeta):
    """A representation of a block matrix that can be split into submatrices corresponding to
    various combinations of fields represented by its original blocks. Python contexts for
    PETSc matrices (of type 'python') will derive from this abstract class. These contexts will be
    used as shells for explicitly assembled matrices in other "native" PETSc formats (e.g. 'aij').

    Any derived class must implement the following methods:

    - :meth:`get_spaces`
    - :meth:`create_index_sets`

    Parameters:
        A: matrix to be wrapped up using this class
        a: bilinear forms corresponding to individual blocks
        kwargs: any application-related context
    """

    # TODO: Many of the required methods are probably not in the list yet.
    DELEGATED_METHODS: typing.ClassVar[list[str]] = [
        "assemblyBegin",
        "assemblyEnd",
        "createVecLeft",
        "createVecRight",
        "diagonalScale",
        "getDiagonal",
        "getDiagonalBlock",
        "mult",
        "multTranspose",
        "norm",
        "setUp",
        "zeroEntries",
    ]

    def __init__(self, A: PETSc.Mat, a: list[list[ufl.Form | None]], **kwargs: dict) -> None:
        self.comm = A.getComm()
        self.kwargs = kwargs

        self._a = a
        self._spaces = None
        self._ISes = None
        self._Mat = A

        # Delegate chosen methods to underlying Mat object
        def _create_callable(method_name):
            def wrapper(mat, *args, **kwargs):
                return getattr(self.Mat, method_name)(*args, **kwargs)

            return wrapper

        delegation_config = getattr(self, "DELEGATED_METHODS", [])
        for method in delegation_config:
            setattr(self, method, _create_callable(method))

        super().__init__()

    @property
    def Mat(self) -> PETSc.Mat:
        """Return the wrapped matrix of "python" type.

        Returns:
            wrapped matrix
        """
        return self._Mat

    @property
    def function_spaces(self) -> tuple[list[FunctionSpace], list[FunctionSpace]]:
        """Return tuple of lists containing function spaces for block rows/columns,
        i.e. test spaces in the first list and trial spaces in the second one.

        Returns:
            test spaces, trial spaces
        """
        if self._spaces is None:
            self._spaces = self.get_spaces()
        return self._spaces

    @abc.abstractmethod
    def get_spaces(self) -> tuple[list[FunctionSpace], list[FunctionSpace]]:
        """Implementation of the method for extraction of function spaces from the associated
        bilinear form.

        Returns:
            test spaces, trial spaces
        """
        pass

    @property
    def ISes(self) -> tuple[list[PETSc.IS], list[PETSc.IS]]:
        """Return tuple of lists containing index sets for block rows and columns respectively.

        Returns:
            row index sets, column index sets
        """
        if self._ISes is None:
            self._ISes = self.create_index_sets()
        return self._ISes

    @abc.abstractmethod
    def create_index_sets(self) -> tuple[list[PETSc.IS], list[PETSc.IS]]:
        """Implementation of the method for extraction of index sets from the associated
        matrix data.

        Returns:
            row index sets, column index sets
        """
        pass

    def create(self, mat: PETSc.Mat) -> None:
        """Prepare index sets and set sizes of the wrapper to match sizes of the wrapped matrix.

        Parameters:
            mat: matrix of type ``"python"``
        """
        # Initialize cache
        spaces = self.function_spaces  # calls self.get_spaces
        index_sets = self.ISes  # calls self.create_index_sets
        assert (len(spaces[0]), len(spaces[1])) == self._block_shape, "Unexpected number of spaces"

        # Check that index sets have correct lengths
        if self._layouts == (
            _MatrixLayout.BLOCK,
            _MatrixLayout.BLOCK,
        ):  # FIXME: Always check!
            assert sum([iset.getSize() for iset in index_sets[0]]) == self.Mat.getSize()[0]
            assert sum([iset.getSize() for iset in index_sets[1]]) == self.Mat.getSize()[1]

        # Set sizes of wrapper 'mat' to match sizes of 'self.Mat'
        mat.setSizes(self.Mat.getSizes(), bsize=self.Mat.getBlockSizes())

    def destroy(self, mat: PETSc.Mat):
        """Destroy created index sets.

        Parameters:
            mat: matrix of type ``"python"``
        """
        for iset_row, iset_col in zip(*self.ISes):
            iset_row.destroy()
            iset_col.destroy()

    def duplicate(self, mat: PETSc.Mat, copy: bool = False) -> None:
        """Duplicate the whole context (involves duplication of the wrapped matrix with all
        index sets), create a new wrapper (:py:class:`petsc4py.PETSc.Mat` object of type 'python')
        and return it.

        Parameters:
            mat: matrix of type ``"python"``
            copy: if ``True``, copy also the values of the wrapped matrix
        """

        newmat_ctx = type(self)(self._Mat.duplicate(copy), self._a, **self.kwargs)
        newmat_ctx._ISes = _copy_index_sets(self._ISes)

        A_splittable = PETSc.Mat().create(comm=self.comm)
        A_splittable.setType("python")
        A_splittable.setPythonContext(newmat_ctx)
        A_splittable.setUp()

        return A_splittable

    def view(self, mat: PETSc.Mat, viewer: PETSc.Viewer | None = None) -> None:
        """View the matrix.

        Parameters:
            mat: matrix of type ``"python"``
            viewer: a viewer instance or ``None`` for the default viewer
        """
        if viewer is None:
            return
        viewer_type = viewer.getType()
        if viewer_type != PETSc.Viewer.Type.ASCII:
            return
        viewer.subtractASCIITab(-1)
        self.Mat.view(viewer)
        viewer.subtractASCIITab(1)

    def __repr__(self):
        return f"{type(self).__name__}(a={self._a!r}, bcs={self._bcs!r})"

    def __str__(self):
        return f"{type(self).__name__}(a={self._a!r}, bcs={self._bcs!r})"


class SplittableMatrixBlock(SplittableMatrixBase):
    """A representation of a block matrix that can be split into submatrices corresponding to
    various combinations of the underlying fields.

    Parameters:
        A: matrix to be wrapped up using this class
        a: bilinear forms corresponding to individual blocks
        kwargs: any application-related context

    Note:
        Check :attr:`SplittableMatrixBase.DELEGATED_METHODS` to reveal the list of methods that are
        automatically delegated to the wrapped :py:class:`petsc4py.PETSc.Mat` object.
    """

    def __init__(self, A: PETSc.Mat, a: list[list[ufl.Form | None]], **kwargs: dict) -> None:
        super().__init__(A, a, **kwargs)

        self._fieldsplit_pc_types = {}

        num_brows = len(a)
        num_bcols = len(a[0])
        ml_brows = _MatrixLayout.BLOCK
        ml_bcols = _MatrixLayout.BLOCK

        self._block_shape = (num_brows, num_bcols)
        self._layouts = (ml_brows, ml_bcols)

    def get_spaces(self) -> tuple[list[FunctionSpace], list[FunctionSpace]]:
        num_brows, num_bcols = self._block_shape
        V = (
            np.full(num_brows, None).tolist(),
            np.full(num_bcols, None).tolist(),
        )
        for i, row in enumerate(self._a):
            assert len(row) == num_bcols, "Incompatible number of blocks in a row"
            for j, a_sub in enumerate(row):
                if a_sub is not None:
                    test_space, trial_space = _extract_spaces(a_sub)
                    if V[0][i] is None:
                        V[0][i] = test_space
                    else:
                        if V[0][i] is not test_space:
                            raise RuntimeError("Mismatched test space detected in a row.")
                    if V[1][j] is None:
                        V[1][j] = trial_space
                    else:
                        if V[1][j] is not trial_space:
                            raise RuntimeError("Mismatched trial space detected in a column.")
        return V

    def create_index_sets(self) -> tuple[list[PETSc.IS], list[PETSc.IS]]:
        # FIXME: Explore in detail: https://github.com/FEniCS/dolfinx/pull/1225
        #   Avoid using LG map and keep index sets with locally owned indices!
        #   Each index set, created as a stride of size `bs * size_local` (bs ... block size)
        #   starting at `rank_offset + block_offset`, must use `PETSc.COMM_SELF`.
        def _get_global_indices(local_is, dofmap, lgmap, blockIS=False):
            size = dofmap.index_map_bs * dofmap.index_map.size_local

            if blockIS:
                owned_is = PETSc.IS(self.comm).createBlock(1, local_is.indices[:size])
                lgmap.applyBlock(owned_is, result=owned_is)
                return owned_is
            else:
                owned_is = PETSc.IS(self.comm).createGeneral(local_is.indices[:size])
                return lgmap.applyIS(owned_is)

        isrows = cpp.la.petsc.create_index_sets(
            [(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in self._spaces[0]]
        )
        iscols = cpp.la.petsc.create_index_sets(
            [(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in self._spaces[1]]
        )
        global_isrows = [
            _get_global_indices(isrow, Vsub.dofmap, self.Mat.getLGMap()[0])
            for isrow, Vsub in zip(isrows, self._spaces[0])
        ]
        global_iscols = [
            _get_global_indices(iscol, Vsub.dofmap, self.Mat.getLGMap()[1])
            for iscol, Vsub in zip(iscols, self._spaces[1])
        ]

        return (global_isrows, global_iscols)

    def _set_fieldsplit_pc_type(self, field_ids: tuple[int], pc_type: PETSc.PC.Type):
        """Store fieldsplit preconditioner type for provided field indices.

        Parameters:
            field_ids: an iterable with field IDs defining the split
            pc_type: type of associated preconditioner
        """
        self._fieldsplit_pc_types[field_ids] = pc_type

    def _get_fieldsplit_pc_type(self, field_ids: tuple[int]) -> PETSc.PC.Type | None:
        """Return stored preconditioner type, or ``None`` if it has not been stored.

        Parameters:
            field_ids: an iterable with field IDs defining the split

        Returns:
            type of associated preconditioner
        """
        return self._fieldsplit_pc_types.get(field_ids, None)

    def createSubMatrix(
        self,
        mat: PETSc.Mat,
        isrow: PETSc.IS,
        iscol: PETSc.IS | None = None,
        submat: PETSc.Mat | None = None,
    ):
        """Create submatrix of the wrapped matrix (based on provided index sets), wrap it using
        a newly created context with corresponding (appropriately shifted) index sets and return
        the result.

        Parameters:
            mat: matrix of type ``"python"``
            isrow: row indices to be extracted
            iscol: column indices to be extracted, same as ``isrow`` if ``None``
            submat: optional resultant matrix
        """

        if submat is not None:
            # NOTE: Repeat call (submatrix already requested in the past), we just need to update
            #       its values from the parent matrix, which may have changed e.g. due within Newton
            #       iterations.
            if submat.getType() == PETSc.Mat.Type.PYTHON:
                subctx = submat.getPythonContext()
                _submat = subctx.Mat if isinstance(subctx, type(self)) else submat
            else:
                _submat = submat
            self.Mat.createSubMatrix(isrow, iscol, _submat)
            # TODO: Is the above line a new assembly? How about using virtual submatrices here?
            return submat

        def _apply_shifting(isets, block_ids, rank_offset):
            """From given list of index sets (*isets*), each of them containing owned global
            indices, pick a subset based on the provided list of increasing block indices
            (*block_ids*). The chosen indices are subsequently shifted by subtracting the given
            rank offset (calculated elsewhere by summing up local ranges of omitted blocks on the
            given rank) and block-global offset calculated by summing up block sizes of any
            intervening blocks.
            """
            pos = -1
            block_offset = 0
            shifted_isets = []
            for idx in block_ids:
                for i in range(pos + 1, idx):
                    block_offset += isets[i].getLocalSize()
                renumbered_indices = isets[idx].indices - rank_offset - block_offset
                shifted_isets.append(PETSc.IS(self.comm).createGeneral(renumbered_indices))
                pos = idx

            return shifted_isets

        def _get_rank_offset(isets, block_ids, spaces):
            rank_offset = 0
            num_blocks = len(isets)
            all_block_ids = set(range(num_blocks))
            xtra_block_ids = all_block_ids.difference(block_ids)
            for i in xtra_block_ids:
                rank_offset += spaces[i].dofmap.index_map.local_range[0]

            return rank_offset

        brow_ids = _find_block_indices(isrow, self.ISes[0])
        rank_offset_0 = _get_rank_offset(self.ISes[0], brow_ids, self._spaces[0])
        shifted_ISes_0 = _apply_shifting(self.ISes[0], brow_ids, rank_offset_0)
        if isrow == iscol and self.ISes[0] == self.ISes[1]:
            bcol_ids = brow_ids
            shifted_ISes_1 = shifted_ISes_0
        else:
            bcol_ids = _find_block_indices(iscol, self.ISes[1])
            rank_offset_1 = _get_rank_offset(self.ISes[1], bcol_ids, self._spaces[1])
            shifted_ISes_1 = _apply_shifting(self.ISes[1], bcol_ids, rank_offset_1)

        submat = self.Mat.createSubMatrix(isrow, iscol)

        # Check type of associated preconditioner to avoid unnecessary wrapping of the submatrix
        pc_type = self._get_fieldsplit_pc_type(brow_ids)
        if self._fieldsplit_pc_types and pc_type != "python":
            return submat

        a = [[self._a[i][j] for j in bcol_ids] for i in brow_ids]
        subctx = SplittableMatrixBlock(submat, a, **self.kwargs)
        subctx._ISes = (shifted_ISes_0, shifted_ISes_1)
        subctx._spaces = (
            [self._spaces[0][i] for i in brow_ids],
            [self._spaces[1][j] for j in bcol_ids],
        )
        assert not subctx._fieldsplit_pc_types  # TODO: Is this correct?

        Asub = PETSc.Mat().create(comm=self.comm)
        Asub.setType("python")
        Asub.setPythonContext(subctx)
        Asub.setUp()

        return Asub


def create_splittable_matrix_block(
    A: PETSc.Mat, a: list[list[ufl.Form | None]], **kwargs: dict
) -> PETSc.Mat:
    """Assemble a splittable block matrix from given data (bilinear form and
    boundary conditions). The returned matrix object of type 'python' is a wrapper for the
    actual matrix of type 'aij'. The wrapped matrix needs to be finalised by calling the
    ``assemble`` method of the returned object.

    Parameters:
        A: matrix to be wrapped up using this class
        a: bilinear forms corresponding to individual blocks
        kwargs: any application-related context

    Returns:
        splittable matrix
    """
    ctx = SplittableMatrixBlock(A, a, **kwargs)
    A_splittable = PETSc.Mat().create(comm=ctx.comm)
    A_splittable.setType("python")
    A_splittable.setPythonContext(ctx)
    A_splittable.setUp()

    return A_splittable


# NOTE:
#   What follows is an attempt to wrap a monolithic matrix in a similar fashion as it has been
#   done with the block matrix. The interface currently works for monolithic matrices with
#   `_MatrixLayout.STRIDE` (see above). The currently missing feature to make this work for matrices
#   with `_MatrixLayout.MIX` is the extraction of index sets for individual fields from the big
#   monolithic matrix.


class SplittableMatrixMonolithic(SplittableMatrixBase):
    """A representation of a monolithic matrix that can be split into submatrices corresponding to
    various combinations of the underlying fields.

    Parameters:
        A: matrix to be wrapped up using this class
        a: bilinear forms corresponding to individual blocks
        kwargs: any application-related context

    Note:
        Use :attr:`SplittableMatrixBase.DELEGATED_METHODS` to reveal the list of methods that are
        automatically delegated to the wrapped :py:class:`petsc4py.PETSc.Mat` object.

    .. todo::

        How to extract index sets for individual fields from a monolithic matrix with
        :attr:`_MatrixLayout.MIX`?
    """

    def __init__(self, A: PETSc.Mat, a: list[list[ufl.Form | None]], **kwargs: dict) -> None:
        super().__init__(A, a, **kwargs)

        # Get block shape and layout of DOFs
        test_space, trial_space = _extract_spaces(a)
        num_brows, ml_brows = _analyse_block_structure(test_space)
        num_bcols, ml_bcols = _analyse_block_structure(trial_space)
        self._test_space = test_space
        self._trial_space = trial_space

        self._block_shape = (num_brows, num_bcols)
        self._layouts = (ml_brows, ml_bcols)

        if min(*self._layouts) == _MatrixLayout.MIX:
            msg = "Wrapping mixed monolithic matrices as splittable matrices not supported"
            raise NotImplementedError(msg)

    def get_spaces(self) -> tuple[list[FunctionSpace], list[FunctionSpace]]:
        num_brows, num_bcols = self._block_shape
        return (
            [self._test_space.sub(i).collapse() for i in range(num_brows)],
            [self._trial_space.sub(j).collapse() for j in range(num_bcols)],
        )

    # FIXME: Implement this!
    def create_index_sets(self) -> tuple[list[PETSc.IS], list[PETSc.IS]]:
        return ([], [])

    def createSubMatrix(
        self,
        mat: PETSc.Mat,
        isrow: PETSc.IS,
        iscol: PETSc.IS | None = None,
        submat: PETSc.Mat | None = None,
    ):
        """Create submatrix of the wrapped matrix (based on provided index sets), wrap it using
        a newly created context with corresponding (appropriately shifted) index sets and return
        the result.

        Parameters:
            mat: matrix of type ``"python"``
            isrow: row indices to be extracted
            iscol: column indices to be extracted, same as ``isrow`` if ``None``
            submat: optional resultant matrix
        """
        if submat is not None:
            # NOTE: Repeat call (submatrix already requested in the past), we just need to update
            #       its values from the parent matrix, which may have changed e.g. due within Newton
            #       iterations.
            self.Mat.createSubMatrix(isrow, iscol, submat.getPythonContext().Mat)
            # TODO: Is the above line a new assembly? How about using virtual submatrices here?
            return submat

        submat = self.Mat.createSubMatrix(isrow, iscol)
        subctx = SplittableMatrixMonolithic(submat, self._a, **self.kwargs)
        subctx._ISes = _copy_index_sets(self._ISes)  # FIXME: Exclude extra terms and renumber!
        subctx._spaces = self._spaces  # FIXME: Exclude extra terms!

        Asub = PETSc.Mat().create(comm=self.comm)
        Asub.setType("python")
        Asub.setPythonContext(subctx)
        Asub.setUp()

        return Asub


def create_splittable_matrix_monolithic(
    A: PETSc.Mat, a: list[list[ufl.Form | None]], **kwargs: dict
) -> PETSc.Mat:
    """Assemble a splittable monolithic matrix from given data (bilinear form and
    boundary conditions). The returned matrix object of type 'python' is a wrapper for the
    actual matrix of type 'aij'. The wrapped matrix needs to be finalised by calling the
    ``assemble`` method of the returned object.

    Parameters:
        A: matrix to be wrapped up using this class
        a: bilinear forms corresponding to individual blocks
        kwargs: any application-related context

    Returns:
        splittable matrix
    """
    ctx = SplittableMatrixMonolithic(A, a, **kwargs)
    A_splittable = PETSc.Mat().create(comm=ctx.comm)
    A_splittable.setType("python")
    A_splittable.setPythonContext(ctx)
    A_splittable.setUp()

    return A_splittable

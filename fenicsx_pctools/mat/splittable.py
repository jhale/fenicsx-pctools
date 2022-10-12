import abc
import enum
from functools import cached_property

import numpy as np

from dolfinx import cpp, fem

from mpi4py import MPI
from petsc4py import PETSc


def _extract_spaces(a):
    """Extract test and trial function spaces from given bilinear form."""

    if isinstance(a, fem.FormMetaClass):
        test_space, trial_space = a.function_spaces
    else:
        test_space, trial_space = map(lambda arg: arg.ufl_function_space(), a.arguments())
    return test_space, trial_space


def _extract_comm(test_space, trial_space):
    """Extract MPI communicators from given couple of function spaces, verify that they are the same
    and return the single communicator.
    """

    comm = test_space.mesh.comm
    assert MPI.Comm.Compare(trial_space.mesh.comm, comm) == MPI.IDENT

    return comm


def _copy_index_sets(isets):
    """Duplicate items contained in given tuple of lists with index sets for block rows/columns."""

    if isets is None:
        return None

    isets_copy = (
        [iset.duplicate() for iset in isets[0]],
        [iset.duplicate() for iset in isets[1]],
    )
    return isets_copy


class MatrixLayout(enum.IntEnum):
    """Type of the layout of matrix entries."""

    MIX = 1
    STRIDE = 2
    BLOCK = 3


def _analyse_block_structure(space):
    """Deduce the number of blocks from the number of subspaces of the given space (return 1 if
    the space have no subspaces) and determine :class:`MatrixLayout` by comparing the number of
    blocks with the dofmap's block size.

    Parameters:
        space (`dfx.FunctionSpace`): space to be analysed

    Returns:
        tuple: number of blocks (subspaces) and :class:`MatrixLayout`

    Note:
        A "truly" mixed function space has components from different subspaces and the block size
        set to 1. The block size is set correctly only for spaces with components from the same
        subspace (e.g. those created with `dfx.VectorFunctionSpace` or `dfx.TensorFunctionSpace`).
    """
    num_blocks = space.element.num_sub_elements or 1  # no/zero subspaces -> single block
    bs = space.dofmap.index_map_bs  # 1 for "truly" mixed function spaces

    if bs < num_blocks:
        dof_layout = MatrixLayout.MIX
    else:
        assert bs == num_blocks
        dof_layout = MatrixLayout.STRIDE

    return num_blocks, dof_layout


def find_block_indices(iset, isets):
    """Verify that given *iset* can be obtained as a subset of *isets* and return corresponding
    block indices (i.e. positions in *isets*).

    Parameters:
        iset (`petsc4py.PETSc.IS`): index set to be found in *isets*
        isets (tuple): iterable containing PETSc IS objects

    Returns:
        list: positions in *isets* marking index sets that need to be concatenated to obtain *iset*

    Raises:
        ValueError: when *iset* cannot be obtained as a subset of *isets*
    """
    block_ids = []
    comm = iset.comm.tompi4py()
    parent_indices = iset.indices
    search_space = {k: v for k, v in enumerate(isets)}

    # NOTE:
    #   In the following loop, we repeatedly iterate over the search space taking one index set
    #   after the other (child IS) and trying to find it at the beginning of the parent IS.
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


class SplittableMatrixBase(object, metaclass=abc.ABCMeta):
    """A representation of a block matrix that can be split into submatrices corresponding to
    various combinations of fields represented by its original blocks. Python contexts for
    PETSc matrices (of type 'python') will derive from this abstract class. These contexts will be
    used as shells for explicitly assembled matrices in other "native" PETSc formats (e.g. 'aij').

    Any derived class must implement the following methods:

    - :meth:`_create_mat_object`
    - :meth:`_create_index_sets`

    """

    # TODO: Many of the required methods are probably not in the list yet.
    DELEGATED_METHODS = [
        "createVecLeft",
        "createVecRight",
        "diagonalScale",
        "getDiagonal",
        "getDiagonalBlock",
        "mult",
        "multTranspose",
        "norm",
        "zeroEntries",
    ]

    def __init__(self, a, bcs, appctx, comm):
        self._a = a
        self._bcs = [] if bcs is None else bcs
        self._appctx = {} if appctx is None else appctx
        self._comm = comm
        self._spaces = None
        self._a_cpp = None
        self._Mat = None
        self._ISes = None

        # Delegate chosen methods to underlying Mat object
        def create_callable(method_name):
            def wrapper(mat, *args, **kwargs):
                return getattr(self.Mat, method_name)(*args, **kwargs)

            return wrapper

        delegation_config = getattr(self, "DELEGATED_METHODS", [])
        for method in delegation_config:
            setattr(self, method, create_callable(method))

        super(SplittableMatrixBase, self).__init__()

    @property
    def appctx(self):
        return self._appctx

    @property
    def comm(self):
        """Return the MPI communicator extracted from attached data or default
        `mpi4py.MPI.COMM_WORLD` if no communicator could have been extracted.
        """
        if self._comm is None:
            self._comm = MPI.COMM_WORLD
        return self._comm

    @property
    def function_spaces(self):
        """Return tuple of lists containing function spaces for block rows/columns,
        i.e. test spaces in the first list and trial spaces in the second one."""
        if self._spaces is None:
            raise ValueError("Function spaces yet to be extracted from associated matrix data")
        return self._spaces

    @cached_property
    def jitted_form(self):
        """Return jitted bilinear form."""
        if self._a_cpp is None:
            self._a_cpp = fem.form(self._a)
        return self._a_cpp

    @cached_property
    def Mat(self):
        """Return the wrapped matrix (*cached* `PETSc.Mat` object)."""
        if self._Mat is None:
            self._Mat = self._create_mat_object()
        return self._Mat

    @abc.abstractmethod
    def _create_mat_object(self):
        """Implementation of the method for creation of the wrapped matrix object."""
        pass

    @cached_property
    def ISes(self):
        """Return tuple of lists containing index sets (*cached* `PETSc.IS` objects) for block
        rows/columns.
        """
        if self._ISes is None:
            self._ISes = self._create_index_sets()
        return self._ISes

    @abc.abstractmethod
    def _create_index_sets(self):
        """Implementation of the method for extraction of index sets from associated matrix data."""
        pass

    def create(self, mat):
        """Prepare the wrapped matrix and index sets (both are *cached* properties), and set sizes
        of the wrapper to match sizes of the wrapped matrix.
        """
        # Initialize cache
        wrapped_mat = self.Mat  # calls self._create_mat_object
        index_sets = self.ISes  # calls self._create_index_sets

        # Check that index sets have correct lengths
        if self._layouts == (
            MatrixLayout.BLOCK,
            MatrixLayout.BLOCK,
        ):  # FIXME: Always check!
            assert sum([iset.getSize() for iset in index_sets[0]]) == wrapped_mat.getSize()[0]
            assert sum([iset.getSize() for iset in index_sets[1]]) == wrapped_mat.getSize()[1]

        # Set sizes of wrapper 'mat' to match sizes of 'wrapped_mat'
        mat.setSizes(wrapped_mat.getSizes(), bsize=wrapped_mat.getBlockSizes())

        # Increment tab level for ASCII output
        wrapped_mat.incrementTabLevel(1, parent=mat)

    def destroy(self, mat):
        """Destroy the wrapped matrix."""
        self.Mat.destroy()
        # TODO: Fieldsplit PC complains when we destroy index sets.
        # for iset in zip(*self.ISes):
        #     iset[0].destroy()
        #     iset[1].destroy()

    def duplicate(self, mat, copy):
        """Duplicate the whole context (involves duplication of the wrapped matrix with all
        index sets), create a new wrapper (`PETSc.Mat` object of type 'python') and return it.
        """

        newmat_ctx = type(self)(self._a, self._bcs, self._comm)
        newmat_ctx._Mat = self._Mat.duplicate(copy)
        newmat_ctx._ISes = _copy_index_sets(self._ISes)

        A = PETSc.Mat().create(comm=self.comm)
        A.setType("python")
        A.setPythonContext(newmat_ctx)
        A.setUp()

        return A

    # def setFromOptions(self, mat):
    #     pass  # TODO: How to treat options like 'opts["pc_fieldsplit_NAME_blocks"] = "0, 1"'

    # def setUp(self, mat):
    #     pass  # TODO: Should we split this into initialize/update?

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        viewer_type = viewer.getType()
        if viewer_type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(f"Object wrapped by {type(self).__name__}:\n")
        viewer.subtractASCIITab(-1)  # TODO: 'incrementTabLevel' command above seems to fail
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
        a (list): list of lists containing bilinear forms corresponding to individual blocks
        bcs (list): list of boundary conditions of type `dolfinx.DirichletBC` (can be None)
        comm (`mpi4py.MPI.Intracomm`): MPI communicator (default: None)

    Note:
        Use ``SplittableMatrixBlock.DELEGATED_METHODS`` to list methods automatically delegated to
        the wrapped `PETSc.Mat` object.
    """

    def __init__(self, a, bcs, appctx=None, comm=None):
        super(SplittableMatrixBlock, self).__init__(a, bcs, appctx, comm)

        # Get block shape and layout of DOFs
        num_brows = len(a)
        num_bcols = len(a[0])
        ml_brows = MatrixLayout.BLOCK
        ml_bcols = MatrixLayout.BLOCK

        self._block_shape = (num_brows, num_bcols)
        self._layouts = (ml_brows, ml_bcols)

        # Get spaces per block rows/columns
        self._spaces = V = (
            np.full(num_brows, None).tolist(),
            np.full(num_bcols, None).tolist(),
        )
        self._comm = None
        for i, row in enumerate(a):
            assert len(row) == num_bcols
            for j, a_sub in enumerate(row):
                if a_sub is not None:
                    test_space, trial_space = _extract_spaces(a_sub)
                    if V[0][i] is None:
                        V[0][i] = test_space
                    else:
                        if not V[0][i] is test_space:
                            raise RuntimeError("Mismatched test space for row.")
                    if V[1][j] is None:
                        V[1][j] = trial_space
                    else:
                        if not V[1][j] is trial_space:
                            raise RuntimeError("Mismatched trial space for column.")

                    # Get MPI communicator
                    comm = _extract_comm(test_space, trial_space)
                    if self._comm is None:
                        self._comm = comm
                    assert MPI.Comm.Compare(comm, self._comm) == MPI.IDENT

    def _create_mat_object(self):
        A = fem.petsc.create_matrix_block(self.jitted_form)

        return A

    def _create_index_sets(self):
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

    def assemblyBegin(self, mat, assembly=None):
        fem.petsc.assemble_matrix_block(self.Mat, self.jitted_form, self._bcs, diagonal=1.0)

    def assemblyEnd(self, mat, assembly=None):
        self.Mat.assemble()

    def createSubMatrix(self, mat, isrow, iscol=None, submat=None):
        """Create submatrix of the wrapped matrix (based on provided index sets), wrap it using
        a newly created context with corresponding (appropriately shifted) index sets and return
        the result.
        """

        if submat is not None:
            # NOTE: Repeat call (submatrix already requested in the past), we just need to update
            #       its values from the parent matrix, which may have changed e.g. due within Newton
            #       iterations.
            self.Mat.createSubMatrix(isrow, iscol, submat.getPythonContext().Mat)
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

        brow_ids = find_block_indices(isrow, self.ISes[0])
        rank_offset_0 = _get_rank_offset(self.ISes[0], brow_ids, self._spaces[0])
        shifted_ISes_0 = _apply_shifting(self.ISes[0], brow_ids, rank_offset_0)
        if isrow == iscol and self.ISes[0] == self.ISes[1]:
            bcol_ids = brow_ids
            shifted_ISes_1 = shifted_ISes_0
        else:
            bcol_ids = find_block_indices(iscol, self.ISes[1])
            rank_offset_1 = _get_rank_offset(self.ISes[1], bcol_ids, self._spaces[1])
            shifted_ISes_1 = _apply_shifting(self.ISes[1], bcol_ids, rank_offset_1)

        submat = self.Mat.createSubMatrix(isrow, iscol)
        a = [[self._a[i][j] for j in bcol_ids] for i in brow_ids]
        bcs = None  # TODO: Ensure that boundary conditions have been applied at this stage.
        subctx = SplittableMatrixBlock(a, bcs, self._appctx, self._comm)
        subctx._a_cpp = [[self.jitted_form[i][j] for j in bcol_ids] for i in brow_ids]
        subctx._Mat = submat
        subctx._ISes = (shifted_ISes_0, shifted_ISes_1)
        subctx._spaces = (
            [self._spaces[0][i] for i in brow_ids],
            [self._spaces[1][j] for j in bcol_ids],
        )

        Asub = PETSc.Mat().create(comm=self.comm)
        Asub.setType("python")
        Asub.setPythonContext(subctx)
        Asub.setUp()

        return Asub


def create_splittable_matrix_block(a, bcs=None, appctx=None, comm=None, options_prefix=None):
    """Routine for assembling a splittable block matrix from given data (bilinear form and boundary
    conditions). The returned `PETSc.Mat` object of type 'python' is a wrapper for the actual
    matrix of type 'aij'. The wrapped matrix needs to be finalised by calling the :meth:`assemble`
    method of the returned object.
    """
    ctx = SplittableMatrixBlock(a, bcs, appctx=appctx, comm=comm)
    ctx.Mat.setOptionsPrefix(options_prefix)
    A = PETSc.Mat().create(comm=ctx.comm)
    A.setType("python")
    A.setPythonContext(ctx)  # NOTE: Set sizes (of matrix A) automatically from ctx.
    A.setUp()

    return A


# NOTE:
#   What follows is an attempt to wrap a monolithic matrix in a similar fashion as it has been
#   done with the block matrix. The interface currently works for monolithic matrices with
#   `MatrixLayout.STRIDE` (see above). The currently missing feature to make this work for matrices
#   with `MatrixLayout.MIX` is the extraction of index sets for individual fields from the big
#   monolithic matrix.


class SplittableMatrixMonolithic(SplittableMatrixBase):
    """A representation of a monolithic matrix that can be split into submatrices corresponding to
    various combinations of the underlying fields.

    Parameters:
        a (list): list of lists containing bilinear forms corresponding to individual blocks
        bcs (list): list of boundary conditions of type `dolfinx.DirichletBC` (can be None)
        comm (`mpi4py.MPI.Intracomm`): MPI communicator (default: None)

    Note:
        Use ``SplittableMatrixMonolithic.DELEGATED_METHODS`` to list methods automatically delegated
        to the wrapped `PETSc.Mat` object.

    .. todo::

        How to extract index sets for individual fields from a monolithic matrix with
        :attr:`MatrixLayout.MIX`?
    """

    def __init__(self, a, bcs, appctx=None, comm=None):
        super(SplittableMatrixMonolithic, self).__init__(a, bcs, appctx, comm)

        # Get block shape and layout of DOFs
        test_space, trial_space = _extract_spaces(a)
        num_brows, ml_brows = _analyse_block_structure(test_space)
        num_bcols, ml_bcols = _analyse_block_structure(trial_space)

        self._block_shape = (num_brows, num_bcols)
        self._layouts = (ml_brows, ml_bcols)

        if min(*self._layouts) == MatrixLayout.MIX:
            msg = "Wrapping mixed monolithic matrices as splittable matrices not supported"
            raise NotImplementedError(msg)

        # Store spaces per block rows/columns
        self._spaces = (
            [test_space.sub(i).collapse() for i in range(num_brows)],
            [trial_space.sub(j).collapse() for j in range(num_bcols)],
        )

        # Get MPI communicator
        self._comm = _extract_comm(test_space, trial_space)

    def _create_mat_object(self):
        A = fem.petsc.create_matrix(self.jitted_form)

        return A

    # FIXME: Implement this!
    def _create_index_sets(self):
        return ([], [])

    def assemblyBegin(self, mat, assembly=None):
        fem.petsc.assemble_matrix(self.Mat, self.jitted_form, self._bcs, diagonal=1.0)

    def assemblyEnd(self, mat, assembly=None):
        self.Mat.assemble()

    def createSubMatrix(self, mat, isrow, iscol=None, submat=None):
        if submat is not None:
            # NOTE: Repeat call (submatrix already requested in the past), we just need to update
            #       its values from the parent matrix, which may have changed e.g. due within Newton
            #       iterations.
            self.Mat.createSubMatrix(isrow, iscol, submat.getPythonContext().Mat)
            # TODO: Is the above line a new assembly? How about using virtual submatrices here?

            return submat

        submat = self.Mat.createSubMatrix(isrow, iscol)
        a = self._a  # FIXME: Exclude extra terms!
        bcs = None  # TODO: Ensure that boundary conditions have been applied at this stage.
        subctx = SplittableMatrixMonolithic(a, bcs, self._appctx, self._comm)
        subctx._Mat = submat
        subctx._ISes = _copy_index_sets(self._ISes)  # FIXME: Exclude extra terms and renumber!
        subctx._spaces = self._spaces  # FIXME: Exclude extra terms!

        Asub = PETSc.Mat().create(comm=self.comm)
        Asub.setType("python")
        Asub.setPythonContext(subctx)
        Asub.setUp()

        return Asub


def create_splittable_matrix_monolithic(a, bcs=None, appctx=None, comm=None, options_prefix=None):
    """Routine for assembling a splittable monolithic matrix from given data (bilinear form and
    boundary conditions). The returned `PETSc.Mat` object of type 'python' is a wrapper for the
    actual matrix of type 'aij'. The wrapped matrix needs to be finalised by calling the
    ``assemble`` method of the returned object.
    """
    ctx = SplittableMatrixMonolithic(a, bcs, appctx=appctx, comm=comm)
    ctx.Mat.setOptionsPrefix(options_prefix)
    A = PETSc.Mat().create(comm=ctx.comm)
    A.setType("python")
    A.setPythonContext(ctx)  # NOTE: Set sizes (of matrix A) automatically from ctx.
    A.setUp()

    return A

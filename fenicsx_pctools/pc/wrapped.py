import re
import numpy as np

from petsc4py import PETSc

from fenicsx_pctools.pc.base import PCBase


class WrappedPC(PCBase):
    r"""A python PC object that works with operators wrapped as splittable matrices.
    In fact, this is a wrapper for another PETSc PC object which is normally configurable via
    PETSc options using the extra prefix ``wrapped_``.
    """

    _prefix = "wrapped_"
    needs_python_pmat = True

    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Preconditioner must be of type 'python'")

        parent_pc = pc
        prefix = parent_pc.getOptionsPrefix() or ""
        prefix += self._prefix

        _, P = parent_pc.getOperators()
        Pctx = P.getPythonContext()

        opts = PETSc.Options(prefix)
        isfieldsplit = opts.getString("pc_type") == "fieldsplit"

        def combine_fields(comm, field_ids, isets):
            iset_indices = np.concatenate([isets[idx].indices for idx in field_ids])
            return PETSc.IS(comm).createGeneral(iset_indices)

        comm = parent_pc.comm
        pc = PETSc.PC().create(comm=comm)  # new PC object that can be configured via runtime opts
        pc.incrementTabLevel(1, parent=parent_pc)
        pc.setOptionsPrefix(prefix)
        fs_args = []
        if isfieldsplit:
            pc.setOperators(P, P)
            for key in opts.getAll().keys():
                opt = key.strip()
                if opt.startswith("pc_fieldsplit_") and opt.endswith("_fields"):
                    splitnum = re.findall("[0-9]+", opt)[0]  # get the left most integer
                    opt = f"pc_fieldsplit_{splitnum}_fields"
                    field_ids = re.findall("[0-9]+", opts.getString(opt))
                    field_ids = list(map(int, field_ids))  # str -> int conversion
                    opts.delValue(opt)  # remove the option from the database
                    iset = combine_fields(comm, field_ids, Pctx.ISes[0])
                    fs_args.append((splitnum, iset))
            pc.setFromOptions()
            pc.setFieldSplitIS(*fs_args)
        else:
            pc.setOperators(Pctx.Mat, Pctx.Mat)
            pc.setFromOptions()
        self.pc = pc

    def update(self, pc):
        if self.pc.getType() == "fieldsplit":
            for subksp in self.pc.getFieldSplitSubKSP():
                subpc = subksp.getPC()
                # subpc.setUp()  # TODO: This is not working. Why?
                if subpc.getType() == "python":
                    subctx = subpc.getPythonContext()
                    subctx.update(subpc)

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)

    def view(self, pc, viewer=None):
        super(WrappedPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse:\n")
            viewer.subtractASCIITab(-1)  # TODO: 'incrementTabLevel' command above seems to fail
            self.pc.view(viewer)
            viewer.subtractASCIITab(1)

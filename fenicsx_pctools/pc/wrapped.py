# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import re

from petsc4py import PETSc

import numpy as np

from fenicsx_pctools.pc.base import PCBase


class WrappedPC(PCBase):
    r"""A Python context for preconditioners that are supposed to work with operators wrapped
    as splittable matrices.

    In fact, this is a wrapper for another PETSc PC object which is normally configurable via
    PETSc options using the extra prefix ``"wrapped_"``.

    Parameters:
        pc: preconditioner of type ``"python"``
    """

    _prefix = "wrapped_"
    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC) -> None:
        parent_pc = pc
        prefix = parent_pc.getOptionsPrefix() or ""
        prefix += self._prefix

        A, P = parent_pc.getOperators()
        Pctx = P.getPythonContext()
        Actx = A.getPythonContext() if A.getType() == PETSc.Mat.Type.PYTHON else None

        opts = PETSc.Options(prefix)
        isfieldsplit = opts.getString("pc_type") == "fieldsplit"

        def combine_fields(comm, field_ids, isets):
            iset_indices = np.concatenate([isets[idx].indices for idx in field_ids])
            return PETSc.IS(comm).createGeneral(iset_indices)

        comm = parent_pc.comm
        pc = PETSc.PC().create(comm=comm)  # new PC object that can be configured via runtime opts
        pc.setOptionsPrefix(prefix)
        fs_args = []
        if isfieldsplit:
            pc.setOperators(A, P)
            for key in opts.getAll().keys():
                opt = key.strip()
                if opt.startswith("pc_fieldsplit_") and opt.endswith("_fields"):
                    # Get split number and corresponding index set
                    splitnum = re.findall("[0-9]+", opt)[0]  # get the left most integer
                    opt_fields = f"pc_fieldsplit_{splitnum}_fields"
                    field_ids = re.findall("[0-9]+", opts.getString(opt_fields))
                    field_ids = list(map(int, field_ids))  # str -> int conversion
                    opts.delValue(opt_fields)  # remove the option from the database
                    iset = combine_fields(comm, field_ids, Pctx.ISes[0])
                    fs_args.append((splitnum, iset))
                    # Make the matrix context to remember field_ids and corresponding pc_type
                    opt_pc_type = f"fieldsplit_{splitnum}_pc_type"
                    fieldsplit_pc_type = opts.getString(opt_pc_type)  # defaults to None
                    Pctx._set_fieldsplit_pc_type(tuple(field_ids), fieldsplit_pc_type)
                    if Actx:
                        Actx._set_fieldsplit_pc_type(tuple(field_ids), fieldsplit_pc_type)
            pc.setFromOptions()
            pc.setFieldSplitIS(*fs_args)
        else:
            Pmat = Pctx.Mat
            Amat = Actx.Mat if Actx else A
            pc.setOperators(Amat, Pmat)
            pc.setFromOptions()
        self.pc = pc

    def update(self, pc: PETSc.PC) -> None:
        if self.pc.getType() == "fieldsplit":
            for subksp in self.pc.getFieldSplitSubKSP():
                subpc = subksp.getPC()
                # subpc.setUp()  # TODO: This is not working. Why?
                if subpc.getType() == "python":
                    subctx = subpc.getPythonContext()
                    subctx.update(subpc)

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.pc.apply(x, y)

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        self.pc.applyTranspose(x, y)

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer | None = None) -> None:
        super().view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.subtractASCIITab(-1)
            self.pc.view(viewer)
            viewer.subtractASCIITab(1)

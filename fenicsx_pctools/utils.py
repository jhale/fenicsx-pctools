# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import dolfinx
from petsc4py import PETSc


def functions_to_vec(u: typing.List[dolfinx.fem.Function], x):
    """Copies functions into block vector."""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            u[i].vector.copy(subvec)
            subvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            with x.localForm() as loc:
                loc.array[offset : offset + size_local] = u[i].vector.array_r
            offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def vec_to_functions(x, u: typing.List[dolfinx.fem.Function]):
    """Copies block vector into functions."""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            subvec.copy(u[i].vector)
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] = x.array_r[offset : offset + size_local]
            offset += size_local
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Tools for easier manipulation with DOLFINx and PETSc/petsc4py objects."""

from petsc4py import PETSc

import dolfinx


def functions_to_vec(u: list[dolfinx.fem.Function], x: PETSc.Vec) -> None:
    """Copies functions into block vector.

    Parameters:
        u: list of functions
        x: block vector
    """
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            u[i].x.petsc_vec.copy(subvec)
            subvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].x.petsc_vec.getLocalSize()
            with x.localForm() as loc:
                loc.array[offset : offset + size_local] = u[i].vector.array_r
            offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def vec_to_functions(x: PETSc.Vec, u: list[dolfinx.fem.Function]) -> None:
    """Copies block vector into functions.

    Parameters:
        x: block vector
        u: list of functions
    """
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            subvec.copy(u[i].x.petsc_vec)
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].x.petsc_vec.getLocalSize()
            u[i].x.petsc_vec.array[:] = x.array_r[offset : offset + size_local]
            offset += size_local
            u[i].x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

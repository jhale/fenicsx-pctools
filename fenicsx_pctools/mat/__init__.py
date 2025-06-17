# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Tools for implementation of custom PETSc/petsc4py Mat-like objects."""

from .splittable import (
    SplittableMatrixBase,
    SplittableMatrixBlock,
    SplittableMatrixMonolithic,
    create_splittable_matrix_block,
    create_splittable_matrix_monolithic,
)

__all__ = [
    "SplittableMatrixBase",
    "SplittableMatrixBlock",
    "SplittableMatrixMonolithic",
    "create_splittable_matrix_block",
    "create_splittable_matrix_monolithic",
]

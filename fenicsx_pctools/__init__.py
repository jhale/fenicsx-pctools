# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

__version__ = "0.7.2"

from fenicsx_pctools import mat, pc

# Imports allowing shorthand specification of preconditioners in PETSc config files
from .pc.pcd import PCDPC_vX, PCDPC_vY
from .pc.wrapped import WrappedPC

__all__ = [
    "mat",
    "pc",
    "utils",
    "WrappedPC",
    "PCDPC_vX",
    "PCDPC_vY",
]

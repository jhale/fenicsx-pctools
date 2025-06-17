# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import abc

from petsc4py import PETSc


class PCBase(metaclass=abc.ABCMeta):
    """An abstract base class for creating Python contexts suitable for PETSc PC objects.

    Any derived class must implement the following methods:

    - :meth:`initialize`
    - :meth:`update`
    - :meth:`apply`
    - :meth:`applyTranspose`

    """

    needs_python_amat: bool = False
    """Set this to ``True`` if the system matrix needs to be of type 'python'."""

    needs_python_pmat: bool = False
    """Set this to ``True`` if the preconditioner matrix needs to be of type 'python'."""

    def __init__(self) -> None:
        self.initialized = False
        super().__init__()

    @abc.abstractmethod
    def initialize(self, pc: PETSc.PC) -> None:
        """Initialize preconditioner.

        Parameters:
            pc: preconditioner of type "python"
        """
        pass

    @abc.abstractmethod
    def update(self, pc: PETSc.PC) -> None:
        """Update preconditioner.

        Parameters:
            pc: preconditioner of type "python"
        """
        pass

    @abc.abstractmethod
    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the preconditioner to ``x`` and return the result in ``y``.

        Parameters:
            pc: preconditioner of type "python"
            x: input vector
            y: output vector
        """
        pass

    @abc.abstractmethod
    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        """Apply the transpose of the preconditioner to ``x`` and return the result in ``y``.

        Parameters:
            pc: preconditioner of type "python"
            x: input vector
            y: output vector
        """
        pass

    def setUp(self, pc: PETSc.PC) -> None:
        """Standard method called by PETSc to set up the preconditioner.

        Any subclass should not override this method, but instead implement :meth:`initialize`
        and :meth:`update`. The former is executed when the current method is called for the first
        time, the latter is executed with any other subsequent call. Such approach allows to defer
        the initialization to the latest possible moment, so that the custom preconditioner can be
        set via runtime options.

        Parameters:
            pc: preconditioner of type "python"

        Raises:
            ValueError: if an operator associated with ``pc`` is of a wrong type
        """

        A, P = pc.getOperators()
        A_type, P_type = A.getType(), P.getType()

        if self.needs_python_amat and A_type != PETSc.Mat.Type.PYTHON:
            msg = f"{pc!r} needs system matrix of type 'python', not {A_type}"
            raise ValueError(msg)
        if self.needs_python_pmat and P_type != PETSc.Mat.Type.PYTHON:
            msg = f"{pc!r} needs preconditioner matrix of type 'python', not {P_type}"
            raise ValueError(msg)

        if self.initialized:
            self.update(pc)
        else:
            self.initialize(pc)
            self.initialized = True

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer | None = None) -> None:
        """View the preconditioner.

        Parameters:
            pc: preconditioner of type "python"
            viewer: a viewer instance or ``None`` for the default viewer
        """
        if viewer is None:
            return
        viewer_type = viewer.getType()
        if viewer_type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(f"Custom preconditioner {type(self).__name__}\n")

# Copyright (C) 2021-2023 Rafinex S.à r.l. and Jack S. Hale
#
# This file is part of FEniCSx-pctools (https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.fem import Function
from fenicsx_pctools.pc.base import PCBase


class PCDPCBase(PCBase):
    r"""A base class for various implementations of Pressure-Convection-Diffusion (PCD)
    preconditioner for incompressible Navier-Stokes equations.
    """

    _prefix = "pcd_"
    needs_python_pmat = True

    def initialize(self, pc: PETSc.PC) -> None:
        prefix = pc.getOptionsPrefix() or ""
        prefix += self._prefix

        _, P = pc.getOperators()
        Pctx = P.getPythonContext()

        test_space, trial_space = Pctx.function_spaces
        assert len(test_space) == len(trial_space) == 1
        if test_space[0] != trial_space[0]:
            raise ValueError("Nonmatching pressure test and trial spaces")

        V_p = test_space[0]

        p_tr = ufl.TrialFunction(V_p)
        p_te = ufl.TestFunction(V_p)

        self.ghosted_workvec = fem.petsc.create_vector(
            fem.form(p_te * ufl.dx)
        )  # aux. vector used to apply BCs

        ufl_form_Mp = Pctx.kwargs.get("ufl_form_Mp", None)
        if ufl_form_Mp is None:
            nu = Pctx.kwargs.get("nu", 1.0)
            ufl_form_Mp = (1.0 / nu) * ufl.inner(p_tr, p_te) * ufl.dx
        form_Mp = fem.form(ufl_form_Mp)

        self.Mp = Mp = fem.petsc.create_matrix(form_Mp)
        Mp.setOption(PETSc.Mat.Option.SPD, True)
        Mp.setOptionsPrefix(prefix + "Mp_")
        fem.petsc.assemble_matrix(Mp, form_Mp, bcs=[], diagonal=1.0)
        Mp.assemble()

        # TODO: Is this needed?
        # Mp.setNullSpace(P.getNullSpace())
        # tnullsp = P.getTransposeNullSpace()
        # if tnullsp.handle != 0:
        #     Mp.setTransposeNullSpace(tnullsp)

        self.ksp_Mp = ksp_Mp = PETSc.KSP().create(comm=pc.comm)
        ksp_Mp.setType(PETSc.KSP.Type.PREONLY)
        ksp_Mp.pc.setType(PETSc.PC.Type.LU)
        ksp_Mp.pc.setFactorSolverType("mumps")
        ksp_Mp.setOptionsPrefix(prefix + "Mp_")
        ksp_Mp.setOperators(Mp)
        ksp_Mp.setFromOptions()
        ksp_Mp.setUp()

        ufl_form_Ap = Pctx.kwargs.get("ufl_form_Ap", None)
        if ufl_form_Ap is None:
            ufl_form_Ap = ufl.inner(ufl.grad(p_tr), ufl.grad(p_te)) * ufl.dx
            # TODO: Add stabilization term so we don't need to think about nullspaces anymore?
            # from dolfinx import Constant
            # ufl_form_Ap += Constant(V_p.mesh, 1e-6) * p_tr * p_te * ufl.dx
        self.form_Ap = fem.form(ufl_form_Ap)

        self.bcs_pcd = bcs_pcd = Pctx.kwargs.get("bcs_pcd", [])

        self.Ap = Ap = fem.petsc.create_matrix(self.form_Ap)
        Ap.setOptionsPrefix(prefix + "Ap_")
        fem.petsc.assemble_matrix(Ap, self.form_Ap, bcs=bcs_pcd, diagonal=1.0)
        Ap.assemble()

        if not bcs_pcd:
            Ap.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            # Create default nullspace
            # TODO: Consider an option `nullsp_Ap = Pctx.kwargs.get("nullsp_Ap", None)``
            null_vec = Ap.createVecLeft()
            null_vec.set(1.0)
            null_vec.normalize()
            nullsp_Ap = PETSc.NullSpace().create(vectors=[null_vec], comm=pc.comm)
            assert nullsp_Ap.test(Ap)
            Ap.setNullSpace(nullsp_Ap)
            Ap.setTransposeNullSpace(nullsp_Ap)  # FIXME: Isn't it automatic with SYMMETRIC opt?
        else:
            Ap.setOption(PETSc.Mat.Option.SPD, True)

        self.ksp_Ap = ksp_Ap = PETSc.KSP().create(comm=pc.comm)
        ksp_Ap.setOptionsPrefix(prefix + "Ap_")
        ksp_Ap.setOperators(Ap)
        ksp_Ap.setType(PETSc.KSP.Type.PREONLY)
        ksp_Ap.pc.setType(PETSc.PC.Type.LU)
        ksp_Ap.pc.setFactorSolverType("mumps")
        ksp_Ap.setFromOptions()
        ksp_Ap.setUp()

        ufl_form_Kp = Pctx.kwargs.get("ufl_form_Kp", None)
        if ufl_form_Kp is None:
            v = Pctx.kwargs.get("v", ufl.grad(Function(V_p)))  # defaults to zero vector
            nu = Pctx.kwargs.get("nu", 1.0)
            ds_in = Pctx.kwargs.get("ds_in", None)
            ufl_form_Kp = (1.0 / nu) * ufl.dot(ufl.grad(p_tr), v) * p_te * ufl.dx
            if type(self).__name__ == "PCDPC_vY" and ds_in is not None:
                n = ufl.FacetNormal(ds_in.ufl_domain())
                ufl_form_Kp -= (1.0 / nu) * ufl.dot(v, n) * p_tr * p_te * ds_in
        self.form_Kp = fem.form(ufl_form_Kp)

        self.Kp = Kp = fem.petsc.create_matrix(self.form_Kp)
        Kp.setOptionsPrefix(prefix + "Kp_")
        fem.petsc.assemble_matrix(Kp, self.form_Kp, bcs=[], diagonal=1.0)
        Kp.assemble()

    def __del__(self):
        self.Kp.destroy()
        self.Ap.destroy()
        self.Mp.destroy()
        self.ghosted_workvec.destroy()

    def update(self, pc: PETSc.PC) -> None:
        self.Kp.zeroEntries()
        fem.petsc.assemble_matrix(self.Kp, self.form_Kp, bcs=[], diagonal=1.0)
        self.Kp.assemble()

    def get_work_vecs(self, v: PETSc.Vec, N: int) -> tuple[PETSc.Vec]:
        """Return ``N`` work vectors, all of them duplicated from ``v``.

        Parameters:
            v: vector to be duplicated
            N: number of vectors to be returned

        Returns:
            N copies of vector ``v``
        """
        try:
            vecs = self._work_vecs
            assert len(vecs) == N
        except AttributeError:
            self._work_vecs = vecs = tuple(v.duplicate() for i in range(N))
        except AssertionError:
            raise ValueError("Number of work vectors cannot be changed")
        return vecs

    def bcs_applier(self, x: PETSc.Vec, transpose: bool = False) -> None:
        """Apply subfield BC associated with the discrete pressure Laplacian :math:`A_p`
        to RHS vector ``x``.

        Parameters:
            x: RHS vector
            transpose: if ``True``, apply lifting from transposed :math:`A_p`
        """
        if not self.bcs_pcd:
            return

        if transpose:
            pass  # FIXME: Assure that self.form_Ap is always symmetric or make the transpose!

        # NOTE:
        #   Vectors passed to `self.apply` method are not ghosted, but lifting must be applied
        #   to a ghosted vector.
        y = self.ghosted_workvec
        with y.localForm() as y_local:
            y_local.set(0.0)
        x.copy(result=y)
        y.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        fem.apply_lifting(y, [self.form_Ap], [self.bcs_pcd])
        y.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(y, self.bcs_pcd)
        y.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # # A simple check why/whether it is needed to do the trick which costs 2x copy
        # fem.apply_lifting(x, [self.form_Ap], [self.bcs_pcd])
        # fem.set_bc(x, self.bcs_pcd)
        # x.axpy(-1.0, y)
        # assert x.norm() == 0.0

        y.copy(result=x)  # move truly updated values back to x

    def view(self, pc: PETSc.PC, viewer: PETSc.Viewer | None = None) -> None:
        super().view(pc, viewer)
        viewer.subtractASCIITab(-1)
        viewer.printfASCII("Solver for discrete Laplace operator on the pressure space:\n")
        viewer.subtractASCIITab(-1)
        self.ksp_Ap.view(viewer)
        viewer.subtractASCIITab(1)
        viewer.printfASCII("Solver for pressure mass matrix:\n")
        viewer.subtractASCIITab(-1)
        self.ksp_Mp.view(viewer)
        viewer.subtractASCIITab(1)
        viewer.subtractASCIITab(1)


class PCDPC_vX(PCDPCBase):
    r"""This class implements a modification of PCD variant similar to the one by
    :cite:t:`olshanskii_pressure_2007`, see also :cite:t:`blechta_towards_2019`.
    """

    def initialize(self, pc: PETSc.PC) -> None:
        super().initialize(pc)

    def update(self, pc: PETSc.PC) -> None:
        super().update(pc)

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""This method implements the action of the inverse of the approximate
        Schur complement :math:`-\hat{S}^{-1}`, namely

        .. math::

            y = -M_p^{-1} (I + K_p A_p^{-1}) x,

        where :math:`K_p` denotes the discrete pressure convection matrix, :math:`A_p` denotes the
        discrete Laplace operator on the pressure space and :math:`M_p` is the pressure mass matrix.
        Note that the solve :math:`A_p^{-1} x` is performed with the application of a subfield BC
        on matrix :math:`A_p` and RHS :math:`x`. The subfield BC,
        together with other necessary information, should be provided as keyword arguments to
        the preconditioner matrix of type ``"python"``, e.g.
        :py:class:`fenicsx_pctools.mat.splittable.SplittableMatrixBlock`.
        The expected keyword arguments include the following items:

            + ``"nu"``: kinematic viscosity :math:`\nu` used to scale :math:`M_p` and :math:`K_p`,
              both by :math:`\nu^{-1}`
            + ``"v"``: FE representation of the velocity vector computed as part the current
              nonlinear iterate (used to assemble :math:`K_p`)
            + ``"bcs_pcd"``: homogeneous Dirichlet BC at **inflow** boundary

        .. note::

            It is crucial that the identity term :math:`I x` is not absorbed into the second,
            compact term in the sense

            .. math::

                y = -M_p^{-1} (A_p + K_p) A_p^{-1} x.

            Keeping the identity term separated is important to get a stability with respect to
            the leading Stokes term.

        Parameters:
            pc: preconditioner of type ``"python"``
            x: input vector
            y: output vector
        """

        (z,) = self.get_work_vecs(x, 1)

        x.copy(result=z)  # z = x
        self.bcs_applier(z)  # apply bcs to z
        self.ksp_Ap.solve(z, y)  # y = A_p^{-1} z
        self.Kp.mult(y, z)  # z = K_p y
        z.axpy(1.0, x)  # z = z + x
        self.ksp_Mp.solve(z, y)  # y = M_p^{-1} z
        # TODO: Check the sign!
        y.scale(-1.0)  # y = -y

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        (z,) = self.get_work_vecs(x, 1)

        self.ksp_Mp.solveTranspose(x, y)  # y = M_p^{-T} x
        self.Kp.multTranspose(y, x)  # x = K_p^T y
        self.bcs_applier(x, transpose=True)  # apply bcs to x
        self.ksp_Ap.solveTranspose(x, z)  # z = A_p^{-T} x
        y.axpy(1.0, z)  # y = y + z
        # TODO: Check the sign!
        y.scale(-1.0)  # y = -y


class PCDPC_vY(PCDPCBase):
    r"""This class implements a modification of steady variant of PCD discussed in the book
    by :cite:t:`elman_finite_2014`, see also :cite:t:`blechta_towards_2019`.
    """

    def initialize(self, pc: PETSc.PC) -> None:
        super().initialize(pc)

    def update(self, pc: PETSc.PC) -> None:
        super().update(pc)

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        r"""This method implements the action of the inverse of the approximate
        Schur complement :math:`- \hat{S}^{-1}`, namely

        .. math::

            y = -(I + A_p^{-1} K_p) M_p^{-1} x,

        where :math:`K_p` denotes the discrete pressure convection matrix, :math:`A_p` denotes the
        discrete Laplace operator on the pressure space and :math:`M_p` is the pressure mass matrix.
        Note that the solve :math:`A_p^{-1} x` is performed with the application of a subfield BC
        on matrix :math:`A_p` and RHS :math:`x`. The subfield BC,
        together with other necessary information, should be provided as keyword arguments to
        the preconditioner matrix of type ``"python"``, e.g.
        :py:class:`fenicsx_pctools.mat.splittable.SplittableMatrixBlock`.
        The expected keyword arguments include the following items:

            + ``"nu"``: kinematic viscosity :math:`\nu` used to scale :math:`M_p` and :math:`K_p`,
              both by :math:`\nu^{-1}`
            + ``"v"``: FE representation of the velocity vector computed as part the current
              nonlinear iterate (used to assemble :math:`K_p`)
            + ``"bcs_pcd"``: homogeneous Dirichlet BC at **outflow** boundary
            + ``"ds_in"``: measure used to get a surface integral at inflow boundary that
              contributes to :math:`K_p`

        .. note::

            It is crucial that the identity term :math:`I x` is
            not absorbed into the second, compact term in the sense

            .. math::

                y = -A_p^{-1} (A_p + K_p) M_p^{-1} x.

            Keeping the identity term separated is important to get a stability with respect to
            the leading Stokes term.

        Parameters:
            pc: preconditioner of type ``"python"``
            x: input vector
            y: output vector
        """
        (z, w) = self.get_work_vecs(x, 2)

        # y = -(I + A_p^{-1} K_p) M_p^{-1} x,
        self.ksp_Mp.solve(x, y)  # y = M_p^{-1} x
        self.Kp.mult(y, w)  # w = K_p y
        self.bcs_applier(w)  # apply bcs to w
        self.ksp_Ap.solve(w, z)  # z = A_p^{-1} w
        y.axpy(1.0, z)  # y = y + z
        # FIXME: How is with the sign business?
        y.scale(-1.0)  # y = -y

    def applyTranspose(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec) -> None:
        (z,) = self.get_work_vecs(x, 1)

        x.copy(result=z)  # z = x
        self.bcs_applier(z, transpose=True)  # apply bcs to z
        self.ksp_Ap.solveTranspose(z, y)  # y = A_p^{-T} z
        self.Kp.multTranspose(y, z)  # z = K_p^T y
        z.axpy(1.0, x)  # z = z + x
        self.ksp_Mp.solveTranspose(z, y)  # y = M_p^{-T} z
        # TODO: Check the sign!
        y.scale(-1.0)  # y = -y

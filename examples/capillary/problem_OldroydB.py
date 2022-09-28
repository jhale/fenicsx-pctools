from functools import cached_property

import numpy as np
from problem_NavierStokes import Problem as NavierStokesProblem

import ufl
from dolfinx import fem


class Problem(NavierStokesProblem):

    model_parameters = ("rho", "mu_0", "mu_1", "G_1")

    def parse_options(self, **kwargs):
        # FIXME: Remove the following option as soon as the Leonov model is implemented!
        _model_type = kwargs.pop("_model_type", "linear")
        assert _model_type in ["linear", "nonlinear"]
        self.application_opts["_model_type"] = _model_type

        super().parse_options(**kwargs)

        # Mapping to Navier-Stokes problem options
        self._ns_opts["mu"] = self.mu_0 + self.mu_1

    def Re(self, v_char, x_char):
        """For given characteristic quantities, compute the Reynolds number."""
        mu = self.mu_0 + self.mu_1
        return self.rho * v_char * x_char / mu

    def Wi(self, dgamma_char):
        """For given characteristic quantities, compute the Weissenberg number."""
        relaxtime = self.mu_1 / self.G_1
        return relaxtime * dgamma_char

    def De(self, t_char):
        """For given characteristic quantities, compute the Deborah number."""
        relaxtime = self.mu_1 / self.G_1
        return relaxtime / t_char

    def grad2_xtra(self, A):
        e, r_index, phi_index, z_index = self.e, self.r_index, self.phi_index, self.z_index
        return 2.0 * ufl.sym(
            A[r_index, phi_index] * ufl.outer(e[phi_index], e[phi_index])
            - A[r_index, phi_index] * ufl.outer(e[r_index], e[r_index])
            + (A[r_index, r_index] - A[phi_index, phi_index]) * ufl.outer(e[r_index], e[phi_index])
            - A[phi_index, z_index] * ufl.outer(e[r_index], e[z_index])
            + A[r_index, z_index] * ufl.outer(e[phi_index], e[z_index])
        )

    def T(self, v, p, B1):
        mu_0 = self.coeff("mu_0")
        G_1 = self.coeff("G_1")
        return -p * self.I + 2.0 * mu_0 * self.D(v) + G_1 * (B1 - self.I)

    def rT(self, v, p, B1):
        r = self.coord_r
        mu_0 = self.coeff("mu_0")
        G_1 = self.coeff("G_1")
        return -r * p * self.I + 2.0 * mu_0 * self.rD(v) + r * G_1 * (B1 - self.I)

    @cached_property
    def _mixed_space(self):
        mesh = self.domain.mesh
        family = "P" if mesh.ufl_cell() == ufl.triangle else "Q"

        # FIXME: Set symmetry to True (it's a bit faster)! Not working for Leonov though.
        return [
            ("v", fem.VectorFunctionSpace(mesh, (family, 2), dim=3)),
            ("p", fem.FunctionSpace(mesh, (family, 1))),
            ("B1", fem.TensorFunctionSpace(mesh, (family, 2), shape=(3, 3), symmetry=None)),
        ]  # TODO: Which FE combination?

    @cached_property
    def solution_vars(self):
        functions = []
        for name, subspace in self._mixed_space:
            functions.append(fem.Function(subspace, name=name))

        # Set B1 to identity tensor
        B1 = functions[-1]
        r_index, phi_index, z_index = self.r_index, self.phi_index, self.z_index
        subindex = {
            "rr": B1.ufl_element().extract_subelement_component((r_index, r_index))[0],
            "pp": B1.ufl_element().extract_subelement_component((phi_index, phi_index))[0],
            "zz": B1.ufl_element().extract_subelement_component((z_index, z_index))[0],
        }

        def icond_B1(x):
            values = np.zeros((9, x.shape[1]))
            values[subindex["rr"]] = 1.0
            values[subindex["pp"]] = 1.0
            values[subindex["zz"]] = 1.0

            return values

        B1.interpolate(icond_B1)

        return tuple(functions)

    @cached_property
    def F_form(self):
        v, p, B1 = self.solution_vars
        v_te, p_te, B1_te = self.test_functions

        dx = ufl.dx

        # Volume contributions
        F_v = ufl.inner(self.rT(v, p, B1), self.D_base(v_te)) * dx
        F_v += ufl.inner(self.T(v, p, B1), self.D_xtra(v_te)) * dx
        F_v += ufl.inner(self.coeff("rho") * ufl.dot(self.rgrad(v), v), v_te) * dx

        F_p = -self.rdiv(v) * p_te * dx

        r = self.coord_r
        v_phi = v[self.phi_index]
        mu_1 = self.coeff("mu_1")
        Y1 = lambda B: B - self.I  # noqa E731
        if self.application_opts["_model_type"] == "nonlinear":
            Y1 = lambda B: ufl.dot(B, ufl.dev(B))  # the simplest Leonov model
            # FIXME: Implement new derived class instead of adding options!
        F_B1 = ufl.inner(r * self.coeff("G_1") * Y1(B1), B1_te) * dx
        F_B1 += (
            ufl.inner(
                mu_1 * (r * ufl.dot(self.grad_base(B1), v) + v_phi * self.grad2_xtra(B1)), B1_te
            )
            * dx
        )
        F_B1 -= (
            ufl.inner(mu_1 * (ufl.dot(self.rgrad(v), B1) + ufl.dot(B1, self.rgrad(v).T)), B1_te)
            * dx
        )

        # FIXME: Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            ds_outlet = self.domain.ds("outlet")
            e, r, r_index, z_index = self.e, self.coord_r, self.r_index, self.z_index
            n = ufl.FacetNormal(self.domain.mesh)
            n = n[r_index] * e[r_index] + n[z_index] * e[z_index]
            F_v += (
                -ufl.inner(
                    2.0 * self.coeff("mu_0") * ufl.dot(self.rD(v), n)
                    + r * self.coeff("G_1") * ufl.dot((B1 - self.I), n),
                    v_te,
                )
                * ds_outlet
            )

        return [F_v, F_p, F_B1]

    @cached_property
    def bcs(self):
        bcs = list(super().bcs)

        # FIXME: What about inlet stress BCs?
        if self.application_opts["_model_type"] == "nonlinear":
            # NOTE: Navier-Stokes-like solution is obtained w/o the following condition. Why?
            VB1 = self.function_spaces[-1]
            B1_ic = self.solution_vars[-1]
            B1_inlet = fem.Function(VB1, name="B1_inlet")
            with B1_inlet.vector.localForm() as inlet, B1_ic.vector.localForm() as ic:
                ic.copy(inlet)

            facetdim = self.domain.mesh.topology.dim - 1
            inlet_dofsVB1 = fem.locate_dofs_topological(VB1, facetdim, self._bndry_facets["in"])
            bcs.append(fem.dirichletbc(B1_inlet, inlet_dofsVB1))

        return tuple(bcs)

    @cached_property
    def pcd_forms(self):
        r = self.coord_r
        r_index, z_index = self.r_index, self.z_index
        dx = ufl.dx
        ds_inlet = self.domain.ds("inlet")

        p, q = self.trial_functions[1], self.test_functions[1]
        v = self.solution_vars[0]
        n = self.facet_normal
        mu = self.coeff("mu_0") + self.coeff("mu_1")
        rho = self.coeff("rho")

        ufl_form_Mp = (1.0 / mu) * r * ufl.inner(p, q) * dx
        ufl_form_Ap = r * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
        ufl_form_Kp = (
            (rho / mu) * r * (p.dx(r_index) * v[r_index] + p.dx(z_index) * v[z_index]) * q * dx
        )
        ufl_form_Kp -= (rho / mu) * r * ufl.dot(v, n) * p * q * ds_inlet

        return {
            "ufl_form_Mp": ufl_form_Mp,
            "ufl_form_Ap": ufl_form_Ap,
            "ufl_form_Kp": ufl_form_Kp,
        }

    @cached_property
    def appctx(self):
        appctx = {"bcs_pcd": self.pcd_bcs_outlet}
        appctx.update(self.pcd_forms)
        return appctx

import ufl
import numpy as np

from functools import cached_property
from dolfinx import fem

from problem_NavierStokes import Problem as NavierStokesProblem


class Problem(NavierStokesProblem):
    def parse_options(self, **kwargs):
        # FIXME: Remove the following option as soon as the Leonov model is implemented!
        _model_type = kwargs.pop("_model_type", "linear")
        assert _model_type in ["linear", "nonlinear"]
        self.application_opts["_model_type"] = _model_type

        for prm in [r"\rho", r"\mu_0", r"\mu_1", r"G_1"]:
            self.coeffs[prm] = kwargs.pop(prm)
        if kwargs:
            raise RuntimeError(f"Unused parameters passed to {type(self).__name__}: {kwargs}")

        # Mapping to Navier-Stokes problem options
        self._ns_opts = {
            r"\rho": self.coeffs[r"\rho"],
            r"\mu": self.coeffs[r"\mu_0"] + self.coeffs[r"\mu_1"],
            "bc_outlet": self.application_opts["bc_outlet"],
        }

    def Re(self, v_char, x_char):
        """For given characteristic quantities, compute the Reynolds number."""
        mu = self.coeffs[r"\mu_0"] + self.coeffs[r"\mu_1"]
        return self.coeffs[r"\rho"] * v_char * x_char / mu

    def Wi(self, dgamma_char):
        """For given characteristic quantities, compute the Weissenberg number."""
        relaxtime = self.coeffs[r"\mu_1"] / self.coeffs[r"G_1"]
        return relaxtime * dgamma_char

    def De(self, t_char):
        """For given characteristic quantities, compute the Deborah number."""
        relaxtime = self.coeffs[r"\mu_1"] / self.coeffs[r"G_1"]
        return relaxtime / t_char

    @cached_property
    def mu0(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"\mu_0"])

    @cached_property
    def mu1(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"\mu_1"])

    @cached_property
    def G1(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"G_1"])

    @cached_property
    def rho(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"\rho"])

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
        return -p * self.I + 2.0 * self.mu0 * self.D(v) + self.G1 * (B1 - self.I)

    def rT(self, v, p, B1):
        r = self.coord_r
        return -r * p * self.I + 2.0 * self.mu0 * self.rD(v) + r * self.G1 * (B1 - self.I)

    @cached_property
    def _mixed_space(self):
        domain = self.domain
        family = "P" if domain.mesh.ufl_cell() == ufl.triangle else "Q"

        # FIXME: Set symmetry to True (it's a bit faster)! Not working for Leonov though.
        return [
            ("v", fem.VectorFunctionSpace(domain.mesh, (family, 2), dim=3)),
            ("p", fem.FunctionSpace(domain.mesh, (family, 1))),
            ("B1", fem.TensorFunctionSpace(domain.mesh, (family, 2), shape=(3, 3), symmetry=None)),
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
        F_v += ufl.inner(self.rho * ufl.dot(self.rgrad(v), v), v_te) * dx

        F_p = -self.rdiv(v) * p_te * dx

        r = self.coord_r
        v_phi = v[self.phi_index]
        Y1 = lambda B: B - self.I  # noqa E731
        if self.application_opts["_model_type"] == "nonlinear":
            Y1 = lambda B: ufl.dot(B, ufl.dev(B))  # the simplest Leonov model
            # FIXME: Implement new derived class instead of adding options!
        F_B1 = ufl.inner(r * self.G1 * Y1(B1), B1_te) * dx
        F_B1 += (
            ufl.inner(
                self.mu1 * (r * ufl.dot(self.grad_base(B1), v) + v_phi * self.grad2_xtra(B1)), B1_te
            )
            * dx
        )
        F_B1 -= (
            ufl.inner(self.mu1 * (ufl.dot(self.rgrad(v), B1) + ufl.dot(B1, self.rgrad(v).T)), B1_te)
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
                    2.0 * self.mu0 * ufl.dot(self.rD(v), n)
                    + r * self.G1 * ufl.dot((B1 - self.I), n),
                    v_te,
                )
                * ds_outlet
            )

        return [F_v, F_p, F_B1]

    @cached_property
    def bcs(self):
        Vv = self.function_spaces[0]
        domain = self.domain
        facetdim = domain.mesh.topology.dim - 1

        r_index, phi_index = self.r_index, self.phi_index
        Vv_r = Vv.sub(r_index).collapse()
        Vv_phi = Vv.sub(phi_index).collapse()

        bnd_in = domain.get_boundary_tag("inlet")
        bnd_out = domain.get_boundary_tag("outlet")
        bnd_symm = domain.get_boundary_tag("symmetry")
        bnd_w1 = domain.get_boundary_tag("bwall_hor")
        bnd_w2 = domain.get_boundary_tag("bwall_ver")
        bnd_w3 = domain.get_boundary_tag("cwall")

        facets_in = np.where(domain.mesh_tags_facets.values == bnd_in)[0]
        facets_out = np.where(domain.mesh_tags_facets.values == bnd_out)[0]
        facets_symm = np.where(domain.mesh_tags_facets.values == bnd_symm)[0]
        facets_w1 = np.where(domain.mesh_tags_facets.values == bnd_w1)[0]
        facets_w2 = np.where(domain.mesh_tags_facets.values == bnd_w2)[0]
        facets_w3 = np.where(domain.mesh_tags_facets.values == bnd_w3)[0]

        inlet_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_in)
        outlet_dofsVv_r = fem.locate_dofs_topological((Vv.sub(r_index), Vv_r), facetdim, facets_out)
        symm_dofsVv_r = fem.locate_dofs_topological((Vv.sub(r_index), Vv_r), facetdim, facets_symm)
        w1_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_w1)
        w2_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_w2)
        w3_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_w3)

        bs = Vv.dofmap.index_map_bs
        bs_shifter = np.array([list(range(bs))]).repeat(inlet_dofsVv.shape[0], axis=0).flatten()
        inlet_dofsVv_expanded = bs * inlet_dofsVv.repeat(bs) + bs_shifter
        bs_shifter = np.array([list(range(bs))]).repeat(w3_dofsVv.shape[0], axis=0).flatten()
        w3_dofsVv_expanded = bs * w3_dofsVv.repeat(bs) + bs_shifter

        # Remove duplicated DOFs for consistency
        cornerdofs = np.intersect1d(w1_dofsVv, inlet_dofsVv)
        w1_dofsVv = np.setdiff1d(w1_dofsVv, cornerdofs)

        cornerdofs = np.intersect1d(symm_dofsVv_r[0], inlet_dofsVv_expanded, return_indices=True)[1]
        symm_dofsVv_r[0] = np.delete(symm_dofsVv_r[0], cornerdofs)
        symm_dofsVv_r[1] = np.delete(symm_dofsVv_r[1], cornerdofs)

        cornerdofs = np.hstack(
            (
                np.intersect1d(outlet_dofsVv_r[0], symm_dofsVv_r[0], return_indices=True)[1],
                np.intersect1d(outlet_dofsVv_r[0], w3_dofsVv_expanded, return_indices=True)[1],
            )
        )
        outlet_dofsVv_r[0] = np.delete(outlet_dofsVv_r[0], cornerdofs)
        outlet_dofsVv_r[1] = np.delete(outlet_dofsVv_r[1], cornerdofs)

        walls_dofsVv = np.unique(np.hstack((w1_dofsVv, w2_dofsVv, w3_dofsVv)))

        # Prescribe BCs
        v_zero = fem.Function(Vv, name="v_zero")
        v_inlet = fem.Function(Vv, name="v_inlet")
        v_inlet.interpolate(self.inlet_velocity_profile)
        v_r_zero = fem.Function(Vv_r, name="v_r_zero")

        bcs = [
            fem.DirichletBC(v_inlet, inlet_dofsVv),
            fem.DirichletBC(v_zero, walls_dofsVv),
        ]

        # FIXME: What about inlet stress BCs?
        if self.application_opts["_model_type"] == "nonlinear":
            # NOTE: Navier-Stokes-like solution is obtained w/o the following condition. Why?
            VB1 = self.function_spaces[-1]
            B1_ic = self.solution_vars[-1]
            B1_inlet = fem.Function(VB1, name="B1_inlet")
            with B1_inlet.vector.localForm() as inlet, B1_ic.vector.localForm() as ic:
                ic.copy(inlet)
            inlet_dofsVB1 = fem.locate_dofs_topological(VB1, facetdim, facets_in)
            bcs.append(fem.DirichletBC(B1_inlet, inlet_dofsVB1))

        # NOTE:
        #   There is an issue with DirichletBC constructor if one tries to send it an empty
        #   array with `dtype == numpy.int32`
        try:
            bcs.append(fem.DirichletBC(v_r_zero, symm_dofsVv_r, Vv.sub(r_index)))
        except TypeError as err:
            if symm_dofsVv_r.size == 0 and symm_dofsVv_r.dtype == np.int32:
                symm_dofsVv_r = np.empty((0, 2))  # uses default dtype which is correctly converted
            else:
                raise err

        if self.application_opts["bc_outlet"] == "NoEnd":
            try:  # NOTE: Same as above.
                bcs.append(fem.DirichletBC(v_r_zero, outlet_dofsVv_r, Vv.sub(r_index)))
            except TypeError as err:
                if outlet_dofsVv_r.size == 0 and outlet_dofsVv_r.dtype == np.int32:
                    outlet_dofsVv_r = np.empty((0, 2))
                else:
                    raise err

        # Enforce zero azimuthal velocity
        axisymm_dofsVv_phi = fem.locate_dofs_geometrical(
            (Vv.sub(phi_index), Vv_phi), lambda x: np.full((x.shape[1],), True)
        )
        v_phi_zero = fem.Function(Vv_phi)
        bcs.append(fem.DirichletBC(v_phi_zero, axisymm_dofsVv_phi, Vv.sub(phi_index)))

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
        mu = self.mu0 + self.mu1

        ufl_form_Mp = (1.0 / mu) * r * ufl.inner(p, q) * dx
        ufl_form_Ap = r * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
        ufl_form_Kp = (
            (self.rho / mu) * r * (p.dx(r_index) * v[r_index] + p.dx(z_index) * v[z_index]) * q * dx
        )
        ufl_form_Kp -= (self.rho / mu) * r * ufl.dot(v, n) * p * q * ds_inlet

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

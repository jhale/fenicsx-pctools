import ufl
import numpy as np

from functools import cached_property
from dolfinx import fem

from problem_NavierStokes import Problem as NavierStokesProblem


class Problem(NavierStokesProblem):

    model_parameters = NavierStokesProblem.model_parameters + tuple(["Wi"])

    def parse_options(self, **kwargs):
        super().parse_options(**kwargs)

        # NOTE:
        #   Remember that beta is the ratio of the solvent viscosity over the total viscosity.
        #   The viscosity of the Navier-Stokes fluid in the reduced problem is assumed to match
        #   the value of the total viscosity.
        self._ns_opts["beta"] = 1.0

    def T(self, v, p, B):
        beta = self.constant("beta")
        Wi = self.constant("Wi")
        return -p * self.I + 2.0 * beta * self.D(v) + ((1.0 - beta) / Wi) * B

    @cached_property
    def _mixed_space(self):
        mesh = self.domain.mesh
        family = "P" if mesh.ufl_cell() == ufl.triangle else "Q"
        gdim = mesh.geometry.dim

        # FIXME: Set symmetry to True (it's a bit faster)!
        return [
            ("v", fem.VectorFunctionSpace(mesh, (family, 2), dim=gdim)),
            ("p", fem.FunctionSpace(mesh, (family, 1))),
            ("B", fem.TensorFunctionSpace(mesh, (family, 2), shape=(gdim, gdim), symmetry=None)),
        ]  # TODO: Which FE combination?

    @property
    def tensor_components_mapping(self):
        B_el = self._mixed_space[-1][1].ufl_element()
        value_size = B_el.value_size()
        subindex = {
            "xx": B_el.extract_subelement_component((0, 0))[0],
            "yy": B_el.extract_subelement_component((1, 1))[0],
            "xy": B_el.extract_subelement_component((0, 1))[0],
            "yx": B_el.extract_subelement_component((1, 0))[0],
        }
        # FIXME: If value_size > 4, add components with z-coordinate!

        return subindex, value_size

    @cached_property
    def solution_vars(self):
        functions = []
        for name, subspace in self._mixed_space:
            functions.append(fem.Function(subspace, name=name))

        # Set B to identity tensor
        subindex, value_size = self.tensor_components_mapping

        def icond_B(x):
            values = np.zeros((value_size, x.shape[1]))
            values[subindex["xx"]] = 1.0
            values[subindex["yy"]] = 1.0

            return values

        B = functions[-1]
        B.interpolate(icond_B)

        return tuple(functions)

    @cached_property
    def F_form(self):
        v, p, B = self.solution_vars
        v_te, p_te, B_te = self.test_functions
        dx = ufl.dx

        # Volume contributions
        F_v = ufl.inner(self.T(v, p, B), self.D(v_te)) * dx
        # NOTE: Inertia omitted!
        # F_v += self.constant("Re") * ufl.inner(ufl.dot(ufl.grad(v), v), v_te) * dx

        F_p = -ufl.div(v) * p_te * dx

        F_B = ufl.inner(ufl.dot(ufl.grad(B), v), B_te) * dx
        F_B -= ufl.inner(ufl.dot(ufl.grad(v), B) + ufl.dot(B, ufl.grad(v).T), B_te) * dx
        F_B += ufl.inner((1.0 / self.constant("Wi")) * (B - self.I), B_te) * dx

        # Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            n = self.facet_normal
            ds_outlet = self.domain.ds("outlet")
            F_v += (
                -ufl.inner(
                    2.0 * self.constant("beta") * ufl.dot(self.D(v), n)
                    + ((1.0 - self.constant("beta")) / self.constant("Wi")) * ufl.dot(B, n),
                    v_te,
                )
                * ds_outlet
            )

        return [F_v, F_p, F_B]

    def inlet_stress_profile(self, x):
        subindex, value_size = self.tensor_components_mapping

        values = np.zeros((value_size, x.shape[1]))
        H = self.domain.specific_dimensions[0]
        Wi = self.Wi
        dvdy = -3.0 * x[1] / (H ** 2.0)  # derivative of v(y) = 1.5 * (1.0 - (y / H) ** 2)

        values[subindex["xx"]] = 1.0 + 2.0 * (Wi * dvdy) ** 2.0
        values[subindex["xy"]] = Wi * dvdy
        values[subindex["yx"]] = values[subindex["xy"]]
        values[subindex["yy"]] = np.ones_like(dvdy)

        return values

    @cached_property
    def bcs(self):
        bcs = list(super().bcs)

        VB = self.function_spaces[-1]
        domain = self.domain
        facetdim = domain.mesh.topology.dim - 1
        bnd_in = domain.get_boundary_tag("inlet")
        facets_in = np.where(domain.mesh_tags_facets.values == bnd_in)[0]
        inlet_dofsVB = fem.locate_dofs_topological(VB, facetdim, facets_in)

        B_inlet = fem.Function(VB, name="B_inlet")
        B_inlet.interpolate(self.inlet_stress_profile)

        bcs.append(fem.DirichletBC(B_inlet, inlet_dofsVB))

        return tuple(bcs)

    @cached_property
    def appctx(self):
        return None

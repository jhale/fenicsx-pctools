import ufl
import numpy as np

from functools import cached_property
from petsc4py import PETSc
from dolfinx import fem

from problem_NavierStokes import Problem as NavierStokesProblem


class Problem(NavierStokesProblem):

    model_parameters = NavierStokesProblem.model_parameters + tuple(["Wi"])
    discretization_schemes = ("TH_naive", "TH_stable", "CR_stable")

    def parse_options(self, **kwargs):
        super().parse_options(**kwargs)

        # NOTE:
        #   Remember that beta is the ratio of the solvent viscosity over the total viscosity.
        #   The viscosity of the Navier-Stokes fluid in the reduced problem is assumed to match
        #   the value of the total viscosity.
        self._ns_opts["beta"] = 1.0

        # NOTE: Take only first two letters corresponding to either TH or CR
        self._ns_opts["scheme"] = self.application_opts["scheme"][:2]

        self.counter = 0  # FIXME: Remove this temporary variable (used to plot projected velocity)!

    def T(self, v, p, B):
        beta = self.coeff("beta")
        Wi = self.coeff("Wi")
        return -p * self.I + 2.0 * beta * self.D(v) + ((1.0 - beta) / Wi) * B

    @cached_property
    def _mixed_space(self):
        mesh = self.domain.mesh
        gdim = mesh.geometry.dim

        if self.application_opts["scheme"] == "CR_stable":
            scheme = [
                ("v", fem.VectorFunctionSpace(mesh, ("CR", 1), dim=gdim)),
                ("p", fem.FunctionSpace(mesh, ("DP", 0))),
                ("B", fem.TensorFunctionSpace(mesh, ("DP", 0), shape=(gdim, gdim), symmetry=True)),
            ]  # FIXME: Wrong element for B!
        elif self.application_opts["scheme"] == "TH_stable":
            scheme = [
                ("v", fem.VectorFunctionSpace(mesh, ("P", 2), dim=gdim)),
                ("p", fem.FunctionSpace(mesh, ("P", 1))),
                ("B", fem.TensorFunctionSpace(mesh, ("P", 2), shape=(gdim, gdim), symmetry=True)),
            ]  # FIXME: Wrong element for B!
        else:
            scheme = [
                ("v", fem.VectorFunctionSpace(mesh, ("P", 2), dim=gdim)),
                ("p", fem.FunctionSpace(mesh, ("P", 1))),
                ("B", fem.TensorFunctionSpace(mesh, ("P", 2), shape=(gdim, gdim), symmetry=True)),
            ]

        return scheme

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

        # NOTE: Uncomment to set nonzero initial guess for velocity
        # functions[0].interpolate(self.inlet_velocity_profile)

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
    def projected_velocity(self):
        # space = fem.FunctionSpace(self.domain.mesh, ("RT", 1))
        space = fem.FunctionSpace(self.domain.mesh, ("BDM", 1))
        return fem.Function(space, name="v0")

    def update_projected_velocity(self):
        if self.application_opts["scheme"] != "TH_naive":
            v = self.solution_vars[0]
            v0 = self.projected_velocity

            # NOTE: Simple update from previous nonlinear iteration
            # with v.vector.localForm() as v_loc, v0.vector.localForm() as v0_loc:
            #     v0_loc.array[:] = v_loc.array_r[:]

            # FIXME: Make use of self._projection utils
            space = v0.function_space
            v0_tr = ufl.TrialFunction(space)
            v0_te = ufl.TestFunction(space)
            a = ufl.inner(v0_tr, v0_te) * ufl.dx
            L = ufl.inner(v, v0_te) * ufl.dx

            Amat = fem.assemble_matrix(a)
            Amat.assemble()

            bvec = fem.create_vector(L)
            with bvec.localForm() as bloc:
                bloc.set(0.0)
            fem.assemble_vector(bvec, L)
            bvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            solver = PETSc.KSP().create(self.domain.comm)
            solver.setOptionsPrefix("velocity_projector_")
            solver.setOperators(Amat)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.getPC().setFactorSolverType("mumps")
            solver.setFromOptions()

            solver.solve(bvec, v0.vector)
            v0.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Q = fem.FunctionSpace(self.domain.mesh, ("DP", 0))
            # q = ufl.TestFunction(Q)
            # qvec = fem.assemble_vector(ufl.div(v0) * q * ufl.dx)
            # PETSc.Sys.Print(f"div(v0) ~ {qvec.norm()} (weakly)")

            # from dolfinx.io import XDMFFile
            # mesh = self.domain.mesh
            # mode = "w" if self.counter == 0 else "a"
            # with XDMFFile(mesh.mpi_comm(), "v0.xdmf", mode) as xfile:
            #     mesh.topology.create_connectivity_all()
            #     xfile.write_mesh(mesh)
            #     xfile.write_function(v0, t=self.counter)
            # self.counter += 1

    @cached_property
    def F_form(self):
        v, p, B = self.solution_vars
        v_te, p_te, B_te = self.test_functions
        dx = ufl.dx

        # Volume contributions
        F_v = ufl.inner(self.T(v, p, B), self.D(v_te)) * dx  # NOTE: Stable scheme uses Laplacian!

        # NOTE: Stabilization required by CR element!
        if self.application_opts["scheme"] == "CR_stable":
            h = ufl.FacetArea(self.domain.mesh)
            F_v += (1.0 / h) * ufl.inner(ufl.jump(v), ufl.jump(v_te)) * ufl.dS

        # NOTE: Inertia omitted!
        # F_v += self.coeff("Re") * ufl.inner(ufl.dot(ufl.grad(v), v), v_te) * dx

        F_p = -ufl.div(v) * p_te * dx  # NOTE: Stable scheme uses '+'!

        if self.application_opts["scheme"] == "TH_naive":
            F_B = ufl.inner(ufl.dot(ufl.grad(B), v), B_te) * dx
        else:
            n = self.facet_normal
            v0 = self.projected_velocity
            F_B = ufl.sqrt(ufl.inner(v0("+"), n("+")) ** 2) * ufl.inner(ufl.jump(B), B_te("+")) * ufl.dS
        F_B -= ufl.inner(ufl.dot(ufl.grad(v), B) + ufl.dot(B, ufl.grad(v).T), B_te) * dx
        F_B += ufl.inner((1.0 / self.coeff("Wi")) * (B - self.I), B_te) * dx

        # Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            n = self.facet_normal
            ds_outlet = self.domain.ds("outlet")
            F_v += (
                -ufl.inner(
                    2.0 * self.coeff("beta") * ufl.dot(self.D(v), n)
                    + ((1.0 - self.coeff("beta")) / self.coeff("Wi")) * ufl.dot(B, n),
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
        facetdim = self.domain.mesh.topology.dim - 1
        inlet_dofsVB = fem.locate_dofs_topological(VB, facetdim, self._bndry_facets["in"])

        B_inlet = fem.Function(VB, name="B_inlet")
        B_inlet.interpolate(self.inlet_stress_profile)

        bcs.append(fem.DirichletBC(B_inlet, inlet_dofsVB))

        return tuple(bcs)

    @cached_property
    def appctx(self):
        return None

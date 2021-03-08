import os
import ufl
import numpy as np

from functools import cached_property
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, cpp
from dolfinx.mesh import locate_entities_boundary, create_meshtags


class Problem(object):

    model_name = os.path.splitext(os.path.basename(__file__))[0][8:]

    def __init__(self, domain, **kwargs):
        self.domain = domain

        # self._projection_utils = {}

        # UFL helpers
        self.I = ufl.Identity(domain.mesh.geometry.dim)  # noqa: E741
        self.facet_normal = ufl.FacetNormal(domain.mesh)
        # self.x = ufl.SpatialCoordinate(domain.mesh)

        # Parse application-specific options
        self.application_opts = {}
        bc_outlet = kwargs.pop("bc_outlet", "NoTraction")
        assert bc_outlet in ["NoTraction", "NoEnd"]
        self.application_opts["bc_outlet"] = bc_outlet

        # Keep options for reduced Navier-Stokes (NS) problem as a special variable
        # NOTE: NS problem is solved to get a good initial condition for more complicated models.
        self._ns_opts = self.application_opts.copy()

        # Parse model parameters
        self.coeffs = {}
        self.parse_options(**kwargs)

    # FIXME: Make me abstract! Derived models will override this instead of __init__
    def parse_options(self, **kwargs):
        for prm in [r"\rho", r"\mu"]:
            self.coeffs[prm] = kwargs.pop(prm)
        if kwargs:
            raise RuntimeError(f"Unused parameters passed to {type(self).__name__}: {kwargs}")

        self._ns_opts.update(self.coeffs)  # copy values to pass the sanity check below

    @property
    def reduced_opts_for_NavierStokes(self):
        for prm in [r"\rho", r"\mu"]:
            if prm not in self._ns_opts:
                raise ValueError(f"Parameter '{prm}' not found among reduced Navier-Stokes options")

        return self._ns_opts

    def Re(self, v_char, x_char):
        """For given characteristic quantities, compute the Reynolds number."""
        return self.coeffs[r"\rho"] * v_char * x_char / self.coeffs[r"\mu"]

    def Wi(self, dgamma_char):
        """For given characteristic quantities, compute the Weissenberg number."""
        relaxtime = 0.0
        return relaxtime * dgamma_char

    def De(self, t_char):
        """For given characteristic quantities, compute the Deborah number."""
        relaxtime = 0.0
        return relaxtime / t_char

    @cached_property
    def mu(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"\mu"])

    @cached_property
    def rho(self):
        return fem.Constant(self.domain.mesh, self.coeffs[r"\rho"])

    def D(self, v):
        return ufl.sym(ufl.grad(v))

    def T(self, v, p):
        return -p * self.I + 2.0 * self.mu * self.D(v)

    @cached_property
    def _mixed_space(self):
        domain = self.domain
        family = "P" if domain.mesh.ufl_cell() == ufl.triangle else "Q"

        return [
            ("v", fem.VectorFunctionSpace(domain.mesh, (family, 2), dim=domain.mesh.geometry.dim)),
            ("p", fem.FunctionSpace(domain.mesh, (family, 1))),
        ]

    @property
    def function_names(self):
        return [val[0] for val in self._mixed_space]

    @property
    def function_spaces(self):
        return [val[1] for val in self._mixed_space]

    @property
    def num_dofs(self):
        num_dofs = 0
        for f in self.solution_vars:
            num_dofs += f.vector.getSize()

        return num_dofs

    @cached_property
    def test_functions(self):
        functions = []
        for subspace in self.function_spaces:
            functions.append(ufl.TestFunction(subspace))

        return tuple(functions)

    @cached_property
    def trial_functions(self):
        functions = []
        for subspace in self.function_spaces:
            functions.append(ufl.TrialFunction(subspace))

        return tuple(functions)

    @cached_property
    def solution_vars(self):
        functions = []
        for name, subspace in self._mixed_space:
            functions.append(fem.Function(subspace, name=name))

        return tuple(functions)

    @cached_property
    def F_form(self):
        v, p = self.solution_vars
        v_te, p_te = self.test_functions
        dx = ufl.dx

        # Volume contributions
        F_v = ufl.inner(self.T(v, p), self.D(v_te)) * dx
        # F_v += self.rho * ufl.inner(ufl.dot(ufl.grad(v), v), v_te) * dx  # NOTE: Inertia omitted!

        F_p = -ufl.div(v) * p_te * dx

        # Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            ds_outlet = self.domain.ds("outlet")
            n = self.facet_normal
            F_v += -ufl.inner(2.0 * self.mu * ufl.dot(self.D(v), n), v_te) * ds_outlet

        return [F_v, F_p]

    @cached_property
    def J_form(self):
        J_form = [
            [None for i in range(len(self.solution_vars))] for j in range(len(self.solution_vars))
        ]

        for i in range(len(self.solution_vars)):
            for j in range(len(self.solution_vars)):
                J_form[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        self.F_form[i],
                        self.solution_vars[j],
                        ufl.TrialFunction(self.solution_vars[j].function_space),
                    )
                )
                if J_form[i][j].empty():
                    J_form[i][j] = None

        return J_form

    def inlet_velocity_profile(self, x):
        values = np.zeros((2, x.shape[1]))
        H = self.domain.specific_dimensions[0]
        values[0] = 1.5 * (1.0 - (x[1] / H) ** 2)

        return values

    @cached_property
    def bcs(self):
        Vv = self.function_spaces[0]
        Vv_y = Vv.sub(1).collapse()

        domain = self.domain
        facetdim = domain.mesh.topology.dim - 1

        bnd_in = domain.get_boundary_tag("inlet")
        bnd_out = domain.get_boundary_tag("outlet")
        bnd_symm = domain.get_boundary_tag("symmetry")
        bnd_wall = domain.get_boundary_tag("wall")
        bnd_cylinder = domain.get_boundary_tag("cylinder")

        facets_in = np.where(domain.mesh_tags_facets.values == bnd_in)[0]
        facets_out = np.where(domain.mesh_tags_facets.values == bnd_out)[0]
        facets_symm = np.where(domain.mesh_tags_facets.values == bnd_symm)[0]
        facets_wall = np.where(domain.mesh_tags_facets.values == bnd_wall)[0]
        facets_cylinder = np.where(domain.mesh_tags_facets.values == bnd_cylinder)[0]

        inlet_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_in)
        outlet_dofsVv_y = fem.locate_dofs_topological((Vv.sub(1), Vv_y), facetdim, facets_out)
        symm_dofsVv_y = fem.locate_dofs_topological((Vv.sub(1), Vv_y), facetdim, facets_symm)
        wall_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_wall)
        cylinder_dofsVv = fem.locate_dofs_topological(Vv, facetdim, facets_cylinder)

        def expand_blocksized_dofs(dofs, bs):
            bs_shifter = np.array([list(range(bs))]).repeat(dofs.shape[0], axis=0).flatten()
            dofs_expanded = bs * dofs.repeat(bs) + bs_shifter
            return dofs_expanded

        bs = Vv.dofmap.index_map_bs
        inlet_dofsVv_expanded = expand_blocksized_dofs(inlet_dofsVv, bs)
        wall_dofsVv_expanded = expand_blocksized_dofs(wall_dofsVv, bs)
        cylinder_dofsVv_expanded = expand_blocksized_dofs(cylinder_dofsVv, bs)

        # Remove duplicated DOFs for consistency
        cornerdofs = np.intersect1d(wall_dofsVv, inlet_dofsVv)
        wall_dofsVv = np.setdiff1d(wall_dofsVv, cornerdofs)

        cornerdofs = np.hstack(
            (
                np.intersect1d(symm_dofsVv_y[0], inlet_dofsVv_expanded, return_indices=True)[1],
                np.intersect1d(symm_dofsVv_y[0], cylinder_dofsVv_expanded, return_indices=True)[1],
            )
        )
        symm_dofsVv_y[0] = np.delete(symm_dofsVv_y[0], cornerdofs)
        symm_dofsVv_y[1] = np.delete(symm_dofsVv_y[1], cornerdofs)

        cornerdofs = np.hstack(
            (
                np.intersect1d(outlet_dofsVv_y[0], symm_dofsVv_y[0], return_indices=True)[1],
                np.intersect1d(outlet_dofsVv_y[0], wall_dofsVv_expanded, return_indices=True)[1],
            )
        )
        outlet_dofsVv_y[0] = np.delete(outlet_dofsVv_y[0], cornerdofs)
        outlet_dofsVv_y[1] = np.delete(outlet_dofsVv_y[1], cornerdofs)

        walls_dofsVv = np.unique(np.hstack((wall_dofsVv, cylinder_dofsVv)))

        # Prescribe BCs
        v_zero = fem.Function(Vv, name="v_zero")
        v_inlet = fem.Function(Vv, name="v_inlet")
        v_inlet.interpolate(self.inlet_velocity_profile)
        v_y_zero = fem.Function(Vv_y, name="v_y_zero")

        bcs = [
            fem.DirichletBC(v_inlet, inlet_dofsVv),
            fem.DirichletBC(v_zero, walls_dofsVv),
        ]

        # NOTE:
        #   There is an issue with DirichletBC constructor if one tries to send it an empty
        #   array with `dtype == numpy.int32`
        try:
            bcs.append(fem.DirichletBC(v_y_zero, symm_dofsVv_y, Vv.sub(1)))
        except TypeError as err:
            if symm_dofsVv_y.size == 0 and symm_dofsVv_y.dtype == np.int32:
                symm_dofsVv_y = np.empty((0, 2))  # uses default dtype which is correctly converted
            else:
                raise err

        if self.application_opts["bc_outlet"] == "NoEnd":
            try:  # NOTE: Same as above.
                bcs.append(fem.DirichletBC(v_y_zero, outlet_dofsVv_y, Vv.sub(1)))
            except TypeError as err:
                if outlet_dofsVv_y.size == 0 and outlet_dofsVv_y.dtype == np.int32:
                    outlet_dofsVv_y = np.empty((0, 2))
                else:
                    raise err

        return tuple(bcs)

    @cached_property
    def appctx(self):
        return None

    # def _get_projection_utils(self, tensor_order, element_metadata=None):
    #     if element_metadata is None:
    #         family = "DG" if self.domain.mesh.ufl_cell() == ufl.triangle else "DQ"
    #         degree = 1
    #         element_metadata = (family, degree)
    #     projection_utils = self._projection_utils.setdefault(tensor_order, {})

    #     if not projection_utils:
    #         domain = self.domain
    #         FE = ufl.FiniteElement(element_metadata[0], domain.mesh.ufl_cell(), element_metadata[1])
    #         if tensor_order == 0:
    #             V = fem.FunctionSpace(domain.mesh, FE)
    #             dummy_tensor = 1.0
    #         elif tensor_order == 1:
    #             V = fem.FunctionSpace(domain.mesh, ufl.VectorElement(FE, dim=3))
    #             dummy_tensor = ufl.as_vector(3 * [1.0])
    #         else:
    #             V = fem.FunctionSpace(domain.mesh, ufl.TensorElement(FE, shape=tensor_order * (3,)))
    #             dummy_tensor = 3 * [1.0]
    #             for i in range(tensor_order - 1):
    #                 dummy_tensor = 3 * [dummy_tensor]
    #             dummy_tensor = ufl.as_tensor(dummy_tensor)

    #         r = self.coord_r
    #         q_te, q_tr = ufl.TestFunction(V), ufl.TrialFunction(V)
    #         dummy_rhs_form = fem.form.Form(ufl.inner(dummy_tensor, q_te) * ufl.dx)
    #         projection_form = fem.form.Form(ufl.inner(r * q_tr, q_te) * ufl.dx)
    #         Amat = fem.assemble_matrix(projection_form)
    #         Amat.assemble()

    #         solver = PETSc.KSP().create(self.domain.comm)
    #         solver.setOptionsPrefix(f"projector_TO{tensor_order}_")
    #         solver.setOperators(Amat)
    #         solver.setType("preonly")
    #         solver.getPC().setType("lu")
    #         solver.getPC().setFactorSolverType("mumps")
    #         solver.setFromOptions()

    #         projection_utils["function_space"] = V
    #         projection_utils["solver"] = solver
    #         projection_utils["rhs_vec"] = fem.create_vector(dummy_rhs_form)

    #     return self._projection_utils[tensor_order]

    # def _project(self, q, L):
    #     tensor_order = len(q._ufl_shape)
    #     solver = self._get_projection_utils(tensor_order)["solver"]
    #     bvec = self._get_projection_utils(tensor_order)["rhs_vec"]

    #     with bvec.localForm() as bloc:
    #         bloc.set(0.0)

    #     fem.assemble_vector(bvec, L)
    #     bvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    #     solver.solve(bvec, q.vector)
    #     q.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # @cached_property
    # def _projected_stress(self):
    #     V = self._get_projection_utils(tensor_order=2)["function_space"]
    #     T = fem.Function(V, name="T")
    #     T_te = ufl.TestFunction(V)
    #     projection_rhs = fem.form.Form(ufl.inner(self.rT(*self.solution_vars), T_te) * ufl.dx)

    #     return T, projection_rhs

    # @property
    # def projected_stress(self):
    #     T_h, L = self._projected_stress
    #     self._project(T_h, L)

    #     return T_h

    # @cached_property
    # def _projected_strain_rate(self):
    #     V = self._get_projection_utils(tensor_order=2)["function_space"]
    #     D = fem.Function(V, name="D")
    #     D_te = ufl.TestFunction(V)
    #     v = self.solution_vars[0]
    #     projection_rhs = fem.form.Form(ufl.inner(self.rD(v), D_te) * ufl.dx)

    #     return D, projection_rhs

    # @property
    # def projected_strain_rate(self):
    #     D_h, L = self._projected_strain_rate
    #     self._project(D_h, L)

    #     return D_h

    # @cached_property
    # def _projected_shear_rate(self):
    #     V = self._get_projection_utils(tensor_order=0)["function_space"]
    #     dgamma = fem.Function(V, name="dgamma")
    #     dgamma_te = ufl.TestFunction(V)
    #     v = self.solution_vars[0]
    #     r = self.coord_r
    #     projection_rhs = fem.form.Form(
    #         r * ufl.sqrt(2.0 * ufl.inner(self.D(v), self.D(v))) * dgamma_te * ufl.dx
    #     )

    #     return dgamma, projection_rhs

    # @property
    # def projected_shear_rate(self):
    #     dgamma_h, L = self._projected_shear_rate
    #     self._project(dgamma_h, L)

    #     return dgamma_h

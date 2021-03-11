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
    model_parameters = ("rho", "mu")
    # NOTE:
    #   Model parameters are coefficients that will be set as class attributes. At the same time,
    #   we store the coefficients wrapped as appropriate DOLFINX/UFL objects in a cache; use `coeff`
    #   member function to access the cached objects.

    def __init__(self, domain, **kwargs):
        self.domain = domain

        self._mean_inlet_velocity = 1.0
        self._projection_utils = {}

        # UFL helpers
        self.r_index = r_index = 1
        self.phi_index = 2
        self.z_index = z_index = 0

        self.x = x = ufl.SpatialCoordinate(domain.mesh)
        self.I = ufl.Identity(3)  # noqa: E741
        self.e = e = ufl.unit_vectors(3)
        self.coord_r = x[r_index]
        self.coord_z = x[z_index]

        n = ufl.FacetNormal(domain.mesh)
        self.facet_normal = n[r_index] * e[r_index] + n[z_index] * e[z_index]

        # Parse application-specific options
        self.application_opts = {}
        bc_inlet = kwargs.pop("bc_inlet", "plug")
        assert bc_inlet in ["plug", "parabolic"]
        self.application_opts["bc_inlet"] = bc_inlet

        bc_outlet = kwargs.pop("bc_outlet", "NoTraction")
        assert bc_outlet in ["NoTraction", "NoEnd"]
        self.application_opts["bc_outlet"] = bc_outlet

        # Keep options for reduced Navier-Stokes (NS) problem as a special variable
        # NOTE: NS problem is solved to get a good initial condition for more complicated models.
        self._ns_opts = self.application_opts.copy()

        # Init cache
        self._coeffs = {}  # for UFL coefficients
        self._bndry_facets = {}  # for boundary facets
        self._projection_utils = {}  # for utilities used to get projected stress, shear rate, etc.

        # Parse model parameters
        self.parse_options(**kwargs)

    def __setattr__(self, name, value):
        if name in self.model_parameters:
            constant = self._coeffs.setdefault(name, fem.Constant(self.domain.mesh, value))
            constant.value = value
        return super().__setattr__(name, value)

    # NOTE: Derived models should override this method instead of the constructor.
    def parse_options(self, **kwargs):
        for prm in self.model_parameters:
            value = kwargs.pop(prm, None)
            if value is None:
                raise ValueError(f"Missing parameter '{prm}'")
            setattr(self, prm, value)
        if kwargs:
            raise RuntimeError(f"Unused parameters passed to {type(self).__name__}: {kwargs}")

        for prm in Problem.model_parameters:
            self._ns_opts[prm] = getattr(self, prm, None)

    @property
    def reduced_opts_for_NavierStokes(self):
        for prm in Problem.model_parameters:
            if prm not in self._ns_opts:
                raise ValueError(f"Parameter '{prm}' not found among reduced Navier-Stokes options")

        return self._ns_opts

    @cached_property
    def _sensor_utils(self):
        r_index = self.r_index
        z_index = self.z_index
        domain = self.domain

        tdim = domain.mesh.topology.dim
        facetdim = tdim - 1
        domain.mesh.topology.create_connectivity(facetdim, tdim)  # mark interior/exterior facets

        Rb, Lb, Rc, Lc = domain.specific_dimensions

        def pb_barrel_wall(x):
            halfwidth = 0.2 * Rc
            return np.logical_and(
                np.isclose(x[z_index], Lb - 0.2 * Rb, atol=halfwidth),
                np.isclose(x[r_index], Rb),
            )

        def pb_capillary_wall(x):
            halfwidth = 0.2 * Rc
            return np.logical_and(
                np.isclose(x[z_index], Lb + 0.5 * Lc, atol=halfwidth),
                np.isclose(x[r_index], Rc),
            )

        def pb_capillary_axis(x):
            halfwidth = 0.2 * Rc
            return np.logical_and(
                np.isclose(x[z_index], Lb + 0.5 * Lc, atol=halfwidth),
                np.isclose(x[r_index], 0.0),
            )

        def init_sensor(pb):
            facets = locate_entities_boundary(domain.mesh, facetdim, pb)  # owned bndry facets
            e_to_v = domain.mesh.topology.connectivity(facetdim, 0).array.reshape(-1, 2)
            # NOTE: entity -> vertex map is used to determine ghost entities in `create_meshtags`

            entities = cpp.graph.AdjacencyList_int32(e_to_v[facets])
            values = np.full(facets.shape[0], 1, dtype=np.int32)

            mt = create_meshtags(domain.mesh, facetdim, entities, values)
            ds = ufl.Measure("ds", domain=domain.mesh, subdomain_data=mt, subdomain_id=1)
            size = domain.comm.allreduce(fem.assemble_scalar(1.0 * ds), op=MPI.SUM)

            return {"mt": mt, "ds": ds, "size": size}

        sensor_utils = {
            "bwall": init_sensor(pb_barrel_wall),
            "cwall": init_sensor(pb_capillary_wall),
            "caxis": init_sensor(pb_capillary_axis),
        }

        return sensor_utils

    def get_sensor_utils(self, sensor_id):
        return self._sensor_utils[sensor_id]

    def Re(self, v_char, x_char):
        """For given characteristic quantities, compute the Reynolds number."""
        return self.rho * v_char * x_char / self.mu

    def Wi(self, dgamma_char):
        """For given characteristic quantities, compute the Weissenberg number."""
        relaxtime = 0.0
        return relaxtime * dgamma_char

    def De(self, t_char):
        """For given characteristic quantities, compute the Deborah number."""
        relaxtime = 0.0
        return relaxtime / t_char

    def coeff(self, prm):
        coeff = self._coeffs.get(prm, None)
        if coeff is None:
            raise ValueError(f"Parameter '{prm}' has not been set")

        return coeff

    def grad_base(self, v):
        e, r_index, z_index = self.e, self.r_index, self.z_index
        f = ufl.outer(v.dx(r_index), e[r_index])
        f += ufl.outer(v.dx(z_index), e[z_index])
        return f

    def grad_xtra(self, v):
        e, r_index, phi_index = self.e, self.r_index, self.phi_index
        f = v[r_index] * ufl.outer(e[phi_index], e[phi_index])
        f -= v[phi_index] * ufl.outer(e[r_index], e[phi_index])
        return f

    def D_base(self, v):
        return ufl.sym(self.grad_base(v))

    def D_xtra(self, v):
        return ufl.sym(self.grad_xtra(v))

    def D(self, v):
        r = self.coord_r
        return self.D_base(v) + (1.0 / r) * self.D_xtra(v)

    def rdiv(self, v):
        r = self.coord_r
        r_index, z_index = self.r_index, self.z_index
        return r * (v[r_index].dx(r_index) + v[z_index].dx(z_index)) + v[r_index]

    def rgrad(self, v):
        r = self.coord_r
        return r * self.grad_base(v) + self.grad_xtra(v)

    def rD(self, v):
        r = self.coord_r
        return r * self.D_base(v) + self.D_xtra(v)

    def T(self, v, p):
        return -p * self.I + 2.0 * self.coeff("mu") * self.D(v)

    def rT(self, v, p):
        r = self.coord_r
        return -r * p * self.I + 2.0 * self.coeff("mu") * self.rD(v)

    @cached_property
    def _mixed_space(self):
        mesh = self.domain.mesh
        family = "P" if mesh.ufl_cell() == ufl.triangle else "Q"

        return [
            ("v", fem.VectorFunctionSpace(mesh, (family, 2), dim=3)),
            ("p", fem.FunctionSpace(mesh, (family, 1))),
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
        F_v = ufl.inner(self.rT(v, p), self.D_base(v_te)) * dx
        F_v += ufl.inner(self.T(v, p), self.D_xtra(v_te)) * dx
        F_v += self.coeff("rho") * ufl.inner(ufl.dot(self.rgrad(v), v), v_te) * dx

        F_p = -self.rdiv(v) * p_te * dx

        # Boundary contributions
        if self.application_opts["bc_outlet"] == "NoEnd":
            ds_outlet = self.domain.ds("outlet")
            n = self.facet_normal
            F_v += -ufl.inner(2.0 * self.coeff("mu") * ufl.dot(self.rD(v), n), v_te) * ds_outlet

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

    @property
    def mean_inlet_velocity(self):
        return self._mean_inlet_velocity

    @mean_inlet_velocity.setter
    def mean_inlet_velocity(self, value):
        self._mean_inlet_velocity = float(value)

    def inlet_velocity_profile(self, x):
        inlet_type = self.application_opts["bc_inlet"]
        V_pist = self.mean_inlet_velocity
        values = np.zeros((3, x.shape[1]))
        if inlet_type == "plug":
            values[self.z_index] = V_pist
        elif inlet_type == "parabolic":
            Rb = float(self.domain.specific_dimensions[0])
            values[self.z_index] = 2.0 * V_pist * (1.0 - (x[self.r_index] / Rb) ** 2)
        else:
            raise NotImplementedError(f"Unknow type of inlet velocity profile '{inlet_type}''")

        return values

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

        bf = self._bndry_facets
        bf["in"] = np.where(domain.mesh_tags_facets.values == bnd_in)[0]
        bf["out"] = np.where(domain.mesh_tags_facets.values == bnd_out)[0]
        bf["symm"] = np.where(domain.mesh_tags_facets.values == bnd_symm)[0]
        bf["w1"] = np.where(domain.mesh_tags_facets.values == bnd_w1)[0]
        bf["w2"] = np.where(domain.mesh_tags_facets.values == bnd_w2)[0]
        bf["w3"] = np.where(domain.mesh_tags_facets.values == bnd_w3)[0]

        inlet_dofsVv = fem.locate_dofs_topological(Vv, facetdim, bf["in"])
        outlet_dofsVv_r = fem.locate_dofs_topological((Vv.sub(r_index), Vv_r), facetdim, bf["out"])
        symm_dofsVv_r = fem.locate_dofs_topological((Vv.sub(r_index), Vv_r), facetdim, bf["symm"])
        w1_dofsVv = fem.locate_dofs_topological(Vv, facetdim, bf["w1"])
        w2_dofsVv = fem.locate_dofs_topological(Vv, facetdim, bf["w2"])
        w3_dofsVv = fem.locate_dofs_topological(Vv, facetdim, bf["w3"])

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
            fem.DirichletBC(v_r_zero, symm_dofsVv_r, Vv.sub(r_index)),
        ]

        if self.application_opts["bc_outlet"] == "NoEnd":
            bcs.append(fem.DirichletBC(v_r_zero, outlet_dofsVv_r, Vv.sub(r_index)))

        # Enforce zero azimuthal velocity
        axisymm_dofsVv_phi = fem.locate_dofs_geometrical(
            (Vv.sub(phi_index), Vv_phi), lambda x: np.full((x.shape[1],), True)
        )
        v_phi_zero = fem.Function(Vv_phi)
        bcs.append(fem.DirichletBC(v_phi_zero, axisymm_dofsVv_phi, Vv.sub(phi_index)))

        return tuple(bcs)

    @cached_property
    def pcd_bcs_inlet(self):
        Vp = self.function_spaces[1]
        facetdim = self.domain.mesh.topology.dim - 1
        inlet_dofsVp = fem.locate_dofs_topological(Vp, facetdim, self._bndry_facets["in"])

        return [fem.DirichletBC(fem.Function(Vp), inlet_dofsVp)]

    @cached_property
    def pcd_bcs_outlet(self):
        Vp = self.function_spaces[1]
        facetdim = self.domain.mesh.topology.dim - 1

        outlet_dofsVp = fem.locate_dofs_topological(Vp, facetdim, self._bndry_facets["out"])
        symm_dofsVp = fem.locate_dofs_topological(Vp, facetdim, self._bndry_facets["symm"])
        w3_dofsVp = fem.locate_dofs_topological(Vp, facetdim, self._bndry_facets["w3"])

        cornerdofs = np.intersect1d(outlet_dofsVp, w3_dofsVp)
        outlet_dofsVp = np.setdiff1d(outlet_dofsVp, cornerdofs).reshape((-1, 1))
        cornerdofs = np.intersect1d(outlet_dofsVp, symm_dofsVp)
        outlet_dofsVp = np.setdiff1d(outlet_dofsVp, cornerdofs).reshape((-1, 1))

        return [fem.DirichletBC(fem.Function(Vp), outlet_dofsVp)]

    @cached_property
    def pcd_forms(self):
        r = self.coord_r
        r_index, z_index = self.r_index, self.z_index
        mu = self.coeff("mu")
        rho = self.coeff("rho")
        dx = ufl.dx
        ds_inlet = self.domain.ds("inlet")

        p, q = self.trial_functions[1], self.test_functions[1]
        v = self.solution_vars[0]
        n = self.facet_normal

        ufl_form_Mp = (1.0 / mu) * r * ufl.inner(p, q) * dx
        ufl_form_Ap = r * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
        ufl_form_Kp = (
            (rho / mu)
            * r
            * (p.dx(r_index) * v[r_index] + p.dx(z_index) * v[z_index])
            * q
            * dx
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

    def _get_projection_utils(self, tensor_order, element_metadata=None):
        if element_metadata is None:
            family = "DG" if self.domain.mesh.ufl_cell() == ufl.triangle else "DQ"
            degree = 1
            element_metadata = (family, degree)
        projection_utils = self._projection_utils.setdefault(tensor_order, {})

        if not projection_utils:
            domain = self.domain
            FE = ufl.FiniteElement(element_metadata[0], domain.mesh.ufl_cell(), element_metadata[1])
            if tensor_order == 0:
                V = fem.FunctionSpace(domain.mesh, FE)
                dummy_tensor = 1.0
            elif tensor_order == 1:
                V = fem.FunctionSpace(domain.mesh, ufl.VectorElement(FE, dim=3))
                dummy_tensor = ufl.as_vector(3 * [1.0])
            else:
                V = fem.FunctionSpace(domain.mesh, ufl.TensorElement(FE, shape=tensor_order * (3,)))
                dummy_tensor = 3 * [1.0]
                for i in range(tensor_order - 1):
                    dummy_tensor = 3 * [dummy_tensor]
                dummy_tensor = ufl.as_tensor(dummy_tensor)

            r = self.coord_r
            q_te, q_tr = ufl.TestFunction(V), ufl.TrialFunction(V)
            dummy_rhs_form = fem.form.Form(ufl.inner(dummy_tensor, q_te) * ufl.dx)
            projection_form = fem.form.Form(ufl.inner(r * q_tr, q_te) * ufl.dx)
            Amat = fem.assemble_matrix(projection_form)
            Amat.assemble()

            solver = PETSc.KSP().create(self.domain.comm)
            solver.setOptionsPrefix(f"projector_TO{tensor_order}_")
            solver.setOperators(Amat)
            solver.setType("preonly")
            solver.getPC().setType("lu")
            solver.getPC().setFactorSolverType("mumps")
            solver.setFromOptions()

            projection_utils["function_space"] = V
            projection_utils["solver"] = solver
            projection_utils["rhs_vec"] = fem.create_vector(dummy_rhs_form)

        return self._projection_utils[tensor_order]

    def _project(self, q, L):
        tensor_order = len(q._ufl_shape)
        solver = self._get_projection_utils(tensor_order)["solver"]
        bvec = self._get_projection_utils(tensor_order)["rhs_vec"]

        with bvec.localForm() as bloc:
            bloc.set(0.0)

        fem.assemble_vector(bvec, L)
        bvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        solver.solve(bvec, q.vector)
        q.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    @cached_property
    def _projected_stress(self):
        V = self._get_projection_utils(tensor_order=2)["function_space"]
        T = fem.Function(V, name="T")
        T_te = ufl.TestFunction(V)
        projection_rhs = fem.form.Form(ufl.inner(self.rT(*self.solution_vars), T_te) * ufl.dx)

        return T, projection_rhs

    @property
    def projected_stress(self):
        T_h, L = self._projected_stress
        self._project(T_h, L)

        return T_h

    @cached_property
    def _projected_strain_rate(self):
        V = self._get_projection_utils(tensor_order=2)["function_space"]
        D = fem.Function(V, name="D")
        D_te = ufl.TestFunction(V)
        v = self.solution_vars[0]
        projection_rhs = fem.form.Form(ufl.inner(self.rD(v), D_te) * ufl.dx)

        return D, projection_rhs

    @property
    def projected_strain_rate(self):
        D_h, L = self._projected_strain_rate
        self._project(D_h, L)

        return D_h

    @cached_property
    def _projected_shear_rate(self):
        V = self._get_projection_utils(tensor_order=0)["function_space"]
        dgamma = fem.Function(V, name="dgamma")
        dgamma_te = ufl.TestFunction(V)
        v = self.solution_vars[0]
        r = self.coord_r
        projection_rhs = fem.form.Form(
            r * ufl.sqrt(2.0 * ufl.inner(self.D(v), self.D(v))) * dgamma_te * ufl.dx
        )

        return dgamma, projection_rhs

    @property
    def projected_shear_rate(self):
        dgamma_h, L = self._projected_shear_rate
        self._project(dgamma_h, L)

        return dgamma_h

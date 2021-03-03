import os
import pandas
import pytest
import itertools
import numpy as np
import ufl
import gmsh

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import cpp, fem, MeshTags
from dolfinx.io import XDMFFile
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags

from fenics_pctools.mat.splittable import create_splittable_matrix_block

from gmsh_capillary import model_setter
from snescontext_capillary import SNESContext
from generate_capillary_output import main as generate_output


def _load_problem_module(model_name, module_dir):
    import importlib.util

    module_name = f"problem_{model_name}"
    module_spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(module_dir, f"{module_name}.py")
    )
    problem_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(problem_module)

    return problem_module


def _set_up_solver(problem, opts, options_prefix=None, options_file=None):

    if options_file is not None:
        PETSc.Sys.Print(f"Parsing options from {options_file}")
        opts.prefixPush(options_prefix)
        opts_str = ""
        with open(options_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    opts_str += f" {line}"
        opts_str = opts_str.strip()
        opts.insertString(opts_str)
        opts.prefixPop()

    # Prepare Jacobian matrix (UFL variational form is required in this step)
    Jmat = create_splittable_matrix_block(
        problem.J_form, problem.bcs, problem.appctx, options_prefix=options_prefix
    )

    # Compile each UFL Form into dolfinx Form for better assembly performance
    F_form = fem.assemble._create_cpp_form(problem.F_form)
    J_form = fem.assemble._create_cpp_form(problem.J_form)

    # Set up PDE
    snesctx = SNESContext(F_form, J_form, problem.solution_vars, problem.bcs)

    # Prepare vectors (jitted forms can be used here)
    Fvec = fem.create_vector_block(F_form)
    x0 = fem.create_vector_block(F_form)

    solver = PETSc.SNES().create(problem.domain.comm)
    solver.setFunction(snesctx.F_block, Fvec)
    solver.setJacobian(snesctx.J_block, J=Jmat, P=None)
    solver.setMonitor(snesctx.monitor)
    # solver.setConvergenceTest(snesctx.converged)  # TODO: Prepare custom convergence test!
    solver.setOptionsPrefix(options_prefix)
    solver.setFromOptions()

    return solver, x0, snesctx


def _get_domain_id(param):
    (mesh_type, cell_type), level = param[:2]
    return f"{mesh_type[0]}-{cell_type}-ref{level}-{repr(param[2:])}"


@pytest.fixture(
    params=itertools.product(
        [
            ("unstructured", "triangle"),
            # ("structured", "quad"),
            # ("structured", "triangle"),
            # ("unstructured", "quad"),
        ],
        [0],  # level of refinement
        [4.0],  # ratio Rb_dls = Rb / Rc
        [8.0],  # ratio Lc_dls = Lc / Rc
    ),
    ids=_get_domain_id,
    scope="module",
)
def domain(comm, request):
    (mesh_type, cell_type), level, Rb_dls, Lc_dls = request.param
    recombined = cell_type == "quad"
    structured = mesh_type == "structured"
    size_field = None if structured else 1

    # Barrel (reservoir) dimensions
    Rb = 7.5e-03
    Lb = 4.0 * Rb

    # Capillary dimensions
    Rc = Rb / Rb_dls
    Lc = Rc * Lc_dls

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh_model = None
    domain_name = None
    if comm.rank == 0:
        gmsh_model = gmsh.model
        model_setter(Rb, Lb, Rc, Lc, structured, recombined, level)
        domain_name = gmsh_model.getCurrent()
        PETSc.Sys.Print(f"Preparing mesh for '{domain_name}'")
        if size_field is not None:
            gmsh_model.mesh.field.setAsBackgroundMesh(size_field)
        gmsh_model.mesh.generate(2)
    domain_name = comm.bcast(domain_name, root=0)
    mesh, mts = gmsh_to_dolfin(gmsh.model, 2, prune_z=True, comm=comm)
    gmsh.finalize()

    regions = {}
    map_target_idx = {}
    for target, mt in mts.items():
        regions_dim = regions.setdefault(mt.dim, {})
        regions_dim[target] = mt
    for dim in regions.keys():
        mt_short, keys = merge_meshtags(regions[dim], dim)
        imap = mesh.topology.index_map(dim)
        if imap is None:
            mesh.topology.create_connectivity_all()
        indices = np.arange(0, imap.size_local + imap.num_ghosts)
        values = np.zeros_like(indices, dtype=np.intc)
        mt_full = MeshTags(mesh, dim, indices, values)
        mt_full.values[mt_short.indices] = mt_short.values
        regions[dim].clear()
        regions[dim] = mt_full
        if dim not in map_target_idx:
            map_target_idx[dim] = keys

    class Domain:
        def __init__(self, mesh, regions, map_target_idx, name="unknown"):
            self.mesh = mesh
            self.regions = regions
            self.map_target_idx = map_target_idx
            self.name = name

            self._h = None

        @property
        def comm(self):
            return self.mesh.mpi_comm()

        @property
        def num_vertices(self):
            return self.mesh.topology.index_map(0).size_global

        @property
        def num_cells(self):
            topology = self.mesh.topology
            return topology.index_map(topology.dim).size_global

        @property
        def h(self):
            if self._h is None:
                topology = self.mesh.topology
                num_cells = topology.index_map(topology.dim).size_local
                self._h = cpp.mesh.h(self.mesh, topology.dim, range(num_cells))
            return self._h

        @property
        def h_min(self):
            return self.comm.allreduce(self.h.min(), op=MPI.MIN)

        @property
        def h_max(self):
            return self.comm.allreduce(self.h.max(), op=MPI.MAX)

        @property
        def boundary_regions(self):
            facetdim = self.mesh.topology.dim - 1
            return regions[facetdim]

        def get_boundary_id(self, part):
            facetdim = self.mesh.topology.dim - 1
            return self.map_target_idx[facetdim][part]

        def ds(self, part=None):
            ds = ufl.Measure("ds", subdomain_data=self.boundary_regions, domain=self.mesh)
            if part is not None:
                ds = ds(self.get_boundary_id(part))
            return ds

    domain = Domain(mesh, regions, map_target_idx, name=domain_name)

    # Remember level and specific dimensions for dimensionless analysis and reporting purposes
    domain.level = level
    domain.specific_dimensions = (Rb, Lb, Rc, Lc)

    return domain


_apparent_shear_rates = 10 ** np.linspace(np.log10(2.0), np.log10(300.0), num=4)
_load_cycle = list(map(lambda q: float(q), _apparent_shear_rates))
_fullconfigs = [
    # (
    #     "NavierStokes",
    #     {
    #         r"\rho": 1.0e03,
    #         r"\mu": 1.2e04,
    #         "bc_inlet": "parabolic",
    #         "bc_outlet": "NoEnd",
    #     },
    # ),
    # (
    #     "PowerLaw",
    #     {
    #         r"\rho": 1.0e03,
    #         r"\mu_0": 3.6e05,
    #         r"\mu_8": 1.0e00,
    #         r"\alpha": 7.21,
    #         r"n": 0.3,
    #         "bc_inlet": "parabolic",
    #         "bc_outlet": "NoEnd",
    #     },
    # ),
    (
        "OldroydB",
        {
            r"\rho": 1.0e03,
            r"\mu_0": 2.0e03,
            r"\mu_1": 1.0e04,
            r"G_1": 1.0e07,
            "bc_inlet": "parabolic",
            "bc_outlet": "NoEnd",
            "_model_type": "linear",  # FIXME: Implement the Leonov model and remove this option!
        },
    ),
]


def _get_fullconfig_id(param):
    return f"{param[0]}-{repr(list(param[1].values()))}"


first_run = True


@pytest.mark.parametrize("fullconfig", _fullconfigs, ids=_get_fullconfig_id)
def test_capillary(domain, fullconfig, results_dir, timestamp, request):
    global first_run

    comm = domain.comm
    Rb, Lb, Rc, Lc = domain.specific_dimensions

    def shear_rate_to_velocity(dgamma_app):
        """Converts apparent shear rate to mean velocity value at the inlet."""
        return dgamma_app * (Rc ** 3) / (4.0 * Rb ** 2)

    # Parse configuration
    model_name = fullconfig[0]
    problem_opts = fullconfig[1].copy()  # NOTE: Must copy here to get the same opts for each run!

    # Get mathematical formulation of the discrete problem
    module_dir, module_name = os.path.split(os.path.realpath(request.node.fspath))
    problem_module = _load_problem_module(model_name, module_dir)
    problem = problem_module.Problem(domain, **problem_opts)

    # # NOTE: Uncomment to save mesh with tagged boundary
    # with XDMFFile(comm, f"mesh.xdmf", "w") as f:
    #     domain.mesh.topology.create_connectivity_all()
    #     f.write_mesh(domain.mesh)
    #     # f.write_meshtags(problem.get_sensor_utils("bwall")["mt"])
    #     f.write_meshtags(domain.regions[1])
    # # from dolfinx.io import VTKFile
    # # f = VTKFile(f"mesh.pvd")
    # # f.write(domain.mesh)
    # return

    PETSc.Log.begin()
    PETSc.Sys.Print("")
    opts = PETSc.Options()
    # opts["options_left"] = None
    # opts["mat_mumps_icntl_4"] = 2
    # opts["mat_mumps_icntl_14"] = 500

    petsc_conf = request.config.getoption("petscconf")
    if petsc_conf is None:
        petsc_conf = os.path.join(module_dir, f"petsc_{model_name}_lu.conf")
    petsc_conf = os.path.abspath(petsc_conf)
    assert os.path.isfile(petsc_conf)

    solver, x0, snesctx = _set_up_solver(
        problem, opts, options_prefix="main_solver_", options_file=petsc_conf
    )

    # Navier-Stokes solve is done first to to get a good initial guess
    run_ic_solve = model_name != "NavierStokes"
    if run_ic_solve:
        ns_module = _load_problem_module("NavierStokes", module_dir)
        ns_problem = ns_module.Problem(domain, **problem.reduced_opts_for_NavierStokes)

        ns_petsc_conf = os.path.join(module_dir, "petsc_NavierStokes_lu.conf")
        assert os.path.isfile(petsc_conf)

        ns_solver, ns_x0, ns_snes_ctx = _set_up_solver(
            ns_problem, opts, options_prefix="ns_solver_", options_file=ns_petsc_conf
        )

        dgamma_app = _load_cycle[0]
        ns_problem.mean_inlet_velocity = shear_rate_to_velocity(dgamma_app)
        for bc in ns_problem.bcs:
            if bc.value.name == "v_inlet":
                bc.value.interpolate(ns_problem.inlet_velocity_profile)

        PETSc.Sys.Print("\nSolving Navier-Stokes problem to get an initial guess")
        with PETSc.Log.Stage("Initial NS solve"):
            ns_solver.solve(None, ns_x0)
            ns_x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        SNESContext.vec_to_functions(ns_x0, problem.solution_vars[:2])

    # Update solution vector with the initial guess from solution variables
    SNESContext.functions_to_vec(problem.solution_vars, x0)

    PETSc.Sys.Print(
        f"\nSolving on mesh with {domain.num_cells:g} cells ({problem.num_dofs:g} DOFs)..."
    )
    produced_xdmf = []
    for counter, dgamma_app in enumerate(_load_cycle):
        Vb_mean = shear_rate_to_velocity(dgamma_app)  # expected mean velocity at the inlet

        # Characteristic quantities
        x_char = Rc  # characteristic length
        v_char = (Rb / Rc) ** 2 * Vb_mean  # characteristic velocity
        t_char = Lc / v_char  # residence time for the fluid in the capillary
        dgamma_char = v_char / x_char  # characteristic shear rate

        # Dimensionless numbers
        Re = problem.Re(v_char, x_char)
        Wi = problem.Wi(dgamma_char)
        De = problem.De(t_char)
        assert np.isclose(De, (Rc / Lc) * Wi)

        # Update boundary conditions
        problem.mean_inlet_velocity = Vb_mean
        for bc in problem.bcs:
            if bc.value.name == "v_inlet":
                bc.value.interpolate(problem.inlet_velocity_profile)

        # Solve the problem
        PETSc.Sys.Print(f"\nApparent shear rate: {dgamma_app:g}")
        with PETSc.Log.Stage(f"Nonlinear solve #{counter}"):
            solver.solve(None, x0)
            x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            info_snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
            time_snes = comm.allreduce(info_snes["time"], op=MPI.SUM) / comm.size

        SNESContext.vec_to_functions(x0, problem.solution_vars)
        v_h = problem.solution_vars[0]
        p_h = problem.solution_vars[1]

        # FIXME: Use `dolfinx.function.Expression` for direct evaluation (instead of projections)!
        with PETSc.Log.Stage(f"Projection postprocessing step #{counter}"):
            T_h = problem.projected_stress
            D_h = problem.projected_strain_rate
            dgamma_h = problem.projected_shear_rate

        # Save results
        r = problem.coord_r
        n = problem.facet_normal
        r_index = problem.r_index
        phi_index = problem.phi_index
        z_index = problem.z_index

        su1 = problem.get_sensor_utils("bwall")
        su2 = problem.get_sensor_utils("cwall")
        su3 = problem.get_sensor_utils("caxis")
        integrals = {
            "Q_in": 2.0 * np.pi * ufl.inner(v_h, -r * n) * domain.ds("inlet"),
            "Q_out": 2.0 * np.pi * ufl.inner(v_h, r * n) * domain.ds("outlet"),
            "force": -T_h[z_index, z_index] * r * domain.ds("inlet"),
            "pressure": p_h / su1["size"] * su1["ds"],
            "shstress": -ufl.sym(T_h)[r_index, z_index] / su2["size"] * su2["ds"],
            "nstress_rr": T_h[r_index, r_index] / su2["size"] * su2["ds"],
            "nstress_pp": T_h[phi_index, phi_index] / su2["size"] * su2["ds"],
            "nstress_zz": T_h[z_index, z_index] / su2["size"] * su2["ds"],
            "dgamma": (2.0 * ufl.inner(D_h, D_h)) ** 0.5 / su2["size"] * su2["ds"],
            "v_slip": v_h[z_index] / su2["size"] * su2["ds"],
            "v_axis": v_h[z_index] / su3["size"] * su3["ds"],
        }
        for key, val in integrals.items():
            if counter == 0:
                integrals[key] = fem.form.Form(val)
            integrals[key] = comm.allreduce(fem.assemble_scalar(integrals[key]), op=MPI.SUM)

        filename = f"{os.path.splitext(module_name[5:])[0]}_{model_name}.csv"
        results_file = os.path.join(results_dir, filename)
        results = {
            "timestamp": timestamp,
            "model": model_name,
            "domain": domain.name,
            "Rb": Rb,
            "Lb": Lb,
            "Rc": Rc,
            "Lc": Lc,
            "num_procs": comm.size,
            "num_dofs": problem.num_dofs,
            "num_coredofs": problem.num_dofs / comm.size,
            "num_elements": domain.num_cells,
            "num_vertices": domain.num_vertices,
            "h_min": domain.h_min,
            "h_max": domain.h_max,
            "its_snes": solver.getIterationNumber(),
            "SNESSolve": time_snes,
            "Re": Re,
            "Wi": Wi,
            "De": De,
            "dgamma_app": dgamma_app,
            "dgamma": integrals["dgamma"],
            "nstress_rr": integrals["nstress_rr"],
            "nstress_pp": integrals["nstress_pp"],
            "nstress_zz": integrals["nstress_zz"],
            "shstress": integrals["shstress"],
            "shstress_app": 0.5 * integrals["pressure"] * Rc / Lc,
            "pressure": integrals["pressure"],
            "force": integrals["force"],
            "v_inlet": Vb_mean,
            "v_slip": integrals["v_slip"],
            "v_axis": integrals["v_axis"],
            "Q_app": Vb_mean * np.pi * Rb ** 2,
            "Q_in": integrals["Q_in"],
            "Q_out": integrals["Q_out"],
        }
        for key, val in problem.coeffs.items():
            results[key] = val
        for key, val in problem.application_opts.items():
            results[key] = val

        if comm.rank == 0:
            data = pandas.DataFrame(results, index=[0])
            PETSc.Sys.Print(f"\nSaved data:\n{data.iloc[-1]}\n")
            if request.config.getoption("overwrite"):
                if first_run:
                    mode = "w"
                    header = True
                else:
                    mode = "a"
                    header = False
                first_run = False
            else:
                mode = "a"
                header = not os.path.exists(results_file)

            data.to_csv(results_file, index=False, mode=mode, header=header)
            generate_output(results_file)

        # Save ParaView plots
        if not request.config.getoption("noxdmf"):
            for field in problem.solution_vars + (T_h, D_h, dgamma_h):
                xfile = f"{model_name}_field_{field.name}.xdmf"
                xfile = os.path.join(results_dir, xfile)
                mode = "w" if counter == 0 else "a"
                with XDMFFile(comm, xfile, mode) as f:
                    if counter == 0:
                        produced_xdmf.append(xfile)
                        f.write_mesh(field.function_space.mesh)
                    f.write_function(field, t=counter)
                if comm.rank == 0:
                    PETSc.Sys.Print(f"  + {os.path.abspath(xfile)}")

            # TODO: Prepare and run ParaView scripts here, or right before clean up below?

        # Reset convergence history
        snesctx.reset()

    # Sweep extra outut when running CI pipeline
    if os.getenv("GITLAB_CI") == "true":
        for i, xfile in enumerate(produced_xdmf):
            if i == 0:
                PETSc.Sys.Print("Removed output:")
            os.remove(xfile)
            os.remove(os.path.splitext(xfile)[0] + ".h5")
            PETSc.Sys.Print(f"  - {os.path.abspath(xfile)}")

    # Reset test envorionment
    if opts.hasName("options_left"):
        PETSc.Sys.Print("Cleaning up PETSc options database including unused options!")
    for opt in opts.getAll().keys():
        opts.delValue(opt)
    first_run = True

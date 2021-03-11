import gmsh
import itertools
import numpy as np
import os
import pandas
import pytest
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import cpp, fem, MeshTags
from dolfinx.io import XDMFFile
from dolfinx.common import list_timings, TimingType
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags

from fenics_pctools.mat.splittable import create_splittable_matrix_block

from gmsh_cylinder import model_setter
from snescontext_cylinder import SNESContext
from output_cylinder import main as generate_output


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
        problem.J_form,
        problem.bcs,
        problem.appctx,
        comm=problem.domain.comm,
        options_prefix=options_prefix,
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
    return "H_{}_L_{}_R_{}_level_{}".format(*param)


@pytest.fixture(
    params=itertools.product(
        [2.0],  # channel halfwidth
        [15.0],  # channel halflength
        [1.0],  # cylinder radius
        [0],  # level of refinement
    ),
    ids=_get_domain_id,
    scope="module",
)
def domain(comm, request):
    H, L, R, level = request.param

    # Generate mesh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh_model = None
    domain_name = None
    if comm.rank == 0:
        gmsh_model = gmsh.model
        model_setter(H, L, R, level)
        domain_name = gmsh_model.getCurrent()
        PETSc.Sys.Print(f"Preparing mesh for '{domain_name}'")
        gmsh_model.mesh.generate(2)
    domain_name = comm.bcast(domain_name, root=0)
    mesh, mts = gmsh_to_dolfin(gmsh.model, 2, prune_z=True, comm=comm)
    gmsh.finalize()

    # Prepare mesh tags on all facets
    facetdim = 1
    mts_merged, bndry_tag_map = merge_meshtags(mts, facetdim)
    imap = mesh.topology.index_map(facetdim)
    indices = np.arange(0, imap.size_local + imap.num_ghosts)
    values = np.zeros_like(indices, dtype=np.intc)
    mesh_tags_facets = MeshTags(mesh, facetdim, indices, values)
    mesh_tags_facets.values[mts_merged.indices] = mts_merged.values

    class Domain:
        def __init__(self, mesh, mesh_tags_facets, bndry_tag_map, name="unknown"):
            self.name = name
            self.mesh = mesh
            self.mesh_tags_facets = mesh_tags_facets

            self._bndry_tag_map = bndry_tag_map
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

        def get_boundary_tag(self, label):
            return self._bndry_tag_map[label]

        def ds(self, label=None):
            ds = ufl.Measure("ds", subdomain_data=self.mesh_tags_facets, domain=self.mesh)
            if label is not None:
                ds = ds(self.get_boundary_tag(label))
            return ds

    domain = Domain(mesh, mesh_tags_facets, bndry_tag_map, name=domain_name)

    # Remember level and specific dimensions for dimensionless analysis and reporting purposes
    domain.level = level
    domain.specific_dimensions = (H, L, R)

    return domain


_fullconfigs = {
    "NavierStokes": {"beta": 1.0, "Re": 0.0, "bc_outlet": "NoEnd"},
    "OldroydB": {"beta": 0.59, "Re": 0.0, "Wi": 0.1, "bc_outlet": "NoEnd"},
}


first_run = True


@pytest.mark.parametrize("model_name", ["OldroydB"])
def test_cylinder(domain, model_name, results_dir, timestamp, request):
    global first_run

    comm = domain.comm
    H, L, R = domain.specific_dimensions

    # # NOTE: Uncomment to save mesh with tagged boundary
    # with XDMFFile(comm, "mesh.xdmf", "w") as f:
    #     domain.mesh.topology.create_connectivity_all()
    #     f.write_mesh(domain.mesh)
    #     f.write_meshtags(domain.mesh_tags_facets)
    # # from dolfinx.io import VTKFile
    # # f = VTKFile(f"mesh.pvd")
    # # f.write(domain.mesh)
    # return

    # Get mathematical formulation of the discrete problem
    problem_opts = _fullconfigs[model_name]
    module_dir, module_name = os.path.split(os.path.realpath(request.node.fspath))
    problem_module = _load_problem_module(model_name, module_dir)
    problem = problem_module.Problem(domain, **problem_opts)

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

    PETSc.Sys.Print(
        f"\nSolving on mesh with {domain.num_cells:g} cells ({problem.num_dofs:g} DOFs)..."
    )
    test_cases = np.arange(0.1, 0.8, 0.1) if model_name != "NavierStokes" else [0.0]
    for counter, Wi in enumerate(test_cases):
        problem.Wi = Wi

        with PETSc.Log.Stage(f"{model_name}: nonlinear solve #{counter}"):
            solver.solve(None, x0)
            x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            info_snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
            time_snes = comm.allreduce(info_snes["time"], op=MPI.SUM) / comm.size

        SNESContext.vec_to_functions(x0, problem.solution_vars)

        # FIXME: Use `dolfinx.function.Expression` for direct evaluation (instead of projections)!
        with PETSc.Log.Stage(f"{model_name}: projection postprocessing step #{counter}"):
            T_h = problem.projected_stress

        # Save results
        n = problem.facet_normal
        e_x = ufl.as_vector([1, 0])
        ds = domain.ds
        T = problem.T(*problem.solution_vars)
        integrals = {
            "F_drag": -ufl.inner(2.0 * ufl.dot(T, n), e_x) * ds("cylinder"),
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
            "H": H,
            "L": L,
            "R": R,
            "num_procs": comm.size,
            "num_dofs": problem.num_dofs,
            "num_coredofs": problem.num_dofs / comm.size,
            "num_elements": domain.num_cells,
            "num_vertices": domain.num_vertices,
            "h_min": domain.h_min,
            "h_max": domain.h_max,
            "its_snes": solver.getIterationNumber(),
            "SNESSolve": time_snes,
            "F_drag": integrals["F_drag"],
        }
        for key in problem.model_parameters:
            results[key] = getattr(problem, key)
        for key, val in problem.application_opts.items():
            results[key] = val

        if comm.rank == 0:
            data = pandas.DataFrame(results, index=[0])
            PETSc.Sys.Print(f"\nSaved data:\n{data.iloc[-1]}")
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

        comm.barrier()
        generate_output(results_file)  # savefig must run on all processes (to prevent deadlocks)

        # Save ParaView plots
        if not request.config.getoption("noxdmf"):
            for field in problem.solution_vars + tuple([T_h]):
                xfile = f"{model_name}_field_{field.name}.xdmf"
                xfile = os.path.join(results_dir, xfile)
                mode = "w" if counter == 0 else "a"
                with XDMFFile(comm, xfile, mode) as f:
                    if counter == 0:
                        f.write_mesh(field.function_space.mesh)
                    f.write_function(field, t=counter)
                PETSc.Sys.Print(f"  + {os.path.abspath(xfile)}")

        # Reset convergence history
        snesctx.reset()

    # List timings
    list_timings(comm, [TimingType.wall])

    # Save logs
    logfile = os.path.join(results_dir, f"petsc_cylinder_{comm.size}.log")
    PETSc.Log.view(viewer=PETSc.Viewer.ASCII(logfile, comm=comm))

    # Reset test envorionment
    if opts.hasName("options_left"):
        PETSc.Sys.Print("Cleaning up PETSc options database including unused options!")
    for opt in opts.getAll().keys():
        opts.delValue(opt)
    first_run = True

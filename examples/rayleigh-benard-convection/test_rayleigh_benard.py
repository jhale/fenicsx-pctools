# --------------------------------------------------------------------------------------------------
# Monkeypatch for running tests on ULHPC cluster Iris (https://hpc.uni.lu/systems/iris/)

# NOTE:
#   Some 3rd party libraries (e.g. gmsh) call methods from the built-in `platform` module
#   which can lead to dangerous forking via subprocess, see https://github.com/tqdm/tqdm/pull/1046.
#
#   See also:
#     - https://github.com/tqdm/tqdm/issues/691
#     - https://github.com/h5py/h5py/issues/1079

import socket

if socket.gethostname().startswith("iris"):
    import os
    import platform
    from collections import namedtuple

    uname_result = namedtuple("uname_result", "system node release version machine processor")
    system, node, release, version, machine = os.uname()
    platform._uname_cache = uname_result(system, node, release, version, machine, machine)
# --------------------------------------------------------------------------------------------------


import os

import numpy as np
import pandas
import pytest
from problem_rayleigh_benard import Problem

from dolfinx import fem
from dolfinx.common import TimingType, list_timings
from dolfinx.io import XDMFFile
from fenicsx_pctools.mat.splittable import create_splittable_matrix_block

from mpi4py import MPI
from petsc4py import PETSc


def target_mesh_size():
    # Get analytic formulae for total number of DOFs (for both tetrahedron and hexahedron)
    num_dofs = lambda N: 3 * (2 * N + 1) ** 3 + 2 * (N + 1) ** 3  # noqa: E731
    polycoeffs = [26, 42, 24, 5]
    assert np.polyval(polycoeffs, 10) == num_dofs(10)

    def _target_mesh_size(comm_size, num_coredofs):
        candidates = np.roots(polycoeffs[:-1] + [polycoeffs[-1] - num_coredofs * comm_size])
        N = [int(np.round(N.real)) for N in candidates if np.isreal(N)]
        assert len(N) == 1

        return N[0]

    return _target_mesh_size


@pytest.fixture(name="target_mesh_size", scope="module")
def target_mesh_size_fixture():
    return target_mesh_size()


# @pytest.mark.parametrize("comm_size", [7 * 2 ** n for n in range(4)])
# def test_dofs(comm_size, target_mesh_size):
#     assert comm_size < 2344  # limit for qos-batch
#     num_dofs = lambda N: 3 * (2 * N + 1) ** 3 + 2 * (N + 1) ** 3
#     N = target_mesh_size(comm_size)
#     print(f"{num_dofs(N) * 1e-6} ({comm_size})")
#     print(N)


@pytest.fixture(scope="module")
def problem(comm, target_mesh_size):
    N = target_mesh_size(comm.size, 30000)

    return Problem(comm, N)


first_run = True


@pytest.mark.parametrize("pc_approach", ["PCD-AMG"])
def test_rayleigh_benard(problem, pc_approach, timestamp, results_dir, request):
    global first_run

    comm = problem.comm

    # Set up PETSc options
    PETSc.Log.begin()
    PETSc.Sys.Print("Setting up options...")
    opts = PETSc.Options()
    opts["options_left"] = None

    problem_prefix = "rb_"
    opts.prefixPush(problem_prefix)
    opts["snes_view"] = None
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_monitor"] = None
    opts["snes_converged_reason"] = None
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 25

    # -- linear solver configurations
    opts["ksp_converged_reason"] = None
    hypre_common_settings = {
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_no_CF": None,
        "pc_hypre_boomeramg_coarsen_type": "HMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i",
        "pc_hypre_boomeramg_P_max": 4,
        "pc_hypre_boomeramg_agg_nl": 1,
        "pc_hypre_boomeramg_agg_num_paths": 2,
    }
    if pc_approach == "LU":
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts["wrapped_pc_type"] = "lu"
        opts["wrapped_pc_factor_mat_solver_type"] = "mumps"
    elif pc_approach == "PCD-AMG":
        opts["ksp_type"] = "fgmres"
        opts["ksp_gmres_modifiedgramschmidt"] = None
        # opts["ksp_monitor_true_residual"] = None
        opts["ksp_monitor"] = None
        opts["ksp_max_it"] = 1000
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "fieldsplit"
        opts["pc_fieldsplit_type"] = "multiplicative"
        opts["pc_fieldsplit_0_fields"] = "0, 1"
        opts["pc_fieldsplit_1_fields"] = "2"
        # GMRES on Navier-stokes with fieldsplit PC
        opts.prefixPush("fieldsplit_0_")
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = 1e-2
        opts["ksp_converged_reason"] = None
        # opts["ksp_monitor_true_residual"] = None
        # opts["ksp_monitor"] = None
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "fieldsplit"
        opts["pc_fieldsplit_type"] = "schur"
        opts["pc_fieldsplit_schur_fact_type"] = "lower"
        opts["pc_fieldsplit_0_fields"] = "0"
        opts["pc_fieldsplit_1_fields"] = "1"
        # -- HYPRE on velocity block
        opts["fieldsplit_0_ksp_type"] = "preonly"
        opts["fieldsplit_0_pc_type"] = "python"
        opts["fieldsplit_0_pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts["fieldsplit_0_wrapped_pc_type"] = "hypre"
        for key, val in hypre_common_settings.items():
            opts[f"fieldsplit_0_wrapped_{key}"] = val
        # -- PCD on pressure block
        opts["fieldsplit_1_ksp_type"] = "preonly"
        opts["fieldsplit_1_pc_type"] = "python"
        opts["fieldsplit_1_pc_python_type"] = "fenicsx_pctools.PCDPC_vY"
        # ---- SOR on mass matrix
        opts["fieldsplit_1_pcd_Mp_ksp_type"] = "richardson"
        opts["fieldsplit_1_pcd_Mp_ksp_max_it"] = 2
        opts["fieldsplit_1_pcd_Mp_pc_type"] = "sor"
        # ---- HYPRE on stiffness matrix
        opts["fieldsplit_1_pcd_Ap_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Ap_pc_type"] = "hypre"
        for key, val in hypre_common_settings.items():
            opts[f"fieldsplit_1_pcd_Ap_{key}"] = val
        opts.prefixPop()  # wrapped_
        opts.prefixPop()  # fieldsplit_0_
        # GMRES with HYPRE on temperature block
        opts.prefixPush("fieldsplit_1_")
        opts["ksp_type"] = "gmres"
        opts["ksp_rtol"] = 1e-4
        opts["ksp_converged_reason"] = None
        # opts["ksp_monitor_true_residual"] = None
        # opts["ksp_monitor"] = None
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "hypre"
        for key, val in hypre_common_settings.items():
            opts[f"{key}"] = val
        opts.prefixPop()  # wrapped_
        opts.prefixPop()  # fieldsplit_1_
        opts.prefixPop()  # wrapped_
    opts.prefixPop()  # rb_

    # Set up PDE
    class PDEProblem:
        def __init__(self, F, J, solution_vars, bcs, P=None):
            self.L = F
            self.a = J
            self.a_precon = P
            self.bcs = bcs
            self.solution_vars = solution_vars

        @staticmethod
        def vec_to_functions(x, u):
            offset = 0
            for i in range(len(u)):
                size_local = u[i].vector.getLocalSize()
                u[i].vector.array[:] = x.array_r[offset : offset + size_local]
                u[i].vector.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                offset += size_local

        @staticmethod
        def functions_to_vec(u, x):
            # NOTE: There is a possibility to use `dolfinx.cpp.la.scatter_local_vectors` instead.
            offset = 0
            for i in range(len(u)):
                size_local = u[i].vector.getLocalSize()
                with x.localForm() as loc:
                    loc.array[offset : offset + size_local] = u[i].vector.array_r
                offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        def F_block(self, snes, x, F):
            with F.localForm() as f_local:
                f_local.set(0.0)  # NOTE: f_local includes ghosts

            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            self.vec_to_functions(x, self.solution_vars)

            fem.petsc.assemble_vector_block(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)
            F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        def J_block(self, snes, x, J, P):
            J.zeroEntries()
            if J.getType() != "python":
                fem.assemble_matrix_block(J, self.a, self.bcs, diagonal=1.0)
            J.assemble()
            if self.a_precon is not None:
                P.zeroEntries()
                if P.getType() != "python":
                    fem.assemble_matrix_block(P, self.a_precon, self.bcs, diagonal=1.0)
                P.assemble()

    # Prepare Jacobian matrix (UFL's variational form is required in this step)
    Jmat = create_splittable_matrix_block(
        problem.J_form, problem.bcs, problem.appctx, options_prefix=problem_prefix
    )

    # Set up pressure null space
    null_vec = Jmat.createVecLeft()
    V_v_map = problem.u[0].function_space.dofmap
    V_p_map = problem.u[1].function_space.dofmap
    offset_p = V_v_map.index_map.size_local * V_v_map.index_map_bs
    offset_T = V_p_map.index_map.size_local * V_p_map.index_map_bs
    null_vec.array[offset_p : offset_p + offset_T] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    Jmat.setNullSpace(nsp)

    # Compile each UFL Form into dolfinx Form for better assembly performance
    F_form = fem.form(problem.F_form)
    J_form = fem.form(problem.J_form)
    pdeproblem = PDEProblem(F_form, J_form, problem.u, problem.bcs)

    # Prepare vectors (jitted forms can be used here)
    Fvec = fem.petsc.create_vector_block(F_form)
    x0 = fem.petsc.create_vector_block(F_form)

    solver = PETSc.SNES().create(comm)
    solver.setFunction(pdeproblem.F_block, Fvec)
    solver.setJacobian(pdeproblem.J_block, J=Jmat, P=None)
    solver.setOptionsPrefix(problem_prefix)
    solver.setFromOptions()
    PETSc.Sys.Print("Setup completed")
    PETSc.Sys.Print(f"Solving on mesh with resolution N = {problem.N}...")

    # # Sanity check -- all solution variables should be zero
    # for var in pdeproblem.solution_vars:
    #     with var.vector.localForm() as loc:  # NOTE: local form includes ghosts
    #         assert not np.all(loc.array_r)
    # x0.zeroEntries()  # NOTE: this does not erase ghosts, we need to update them
    # x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    if request.config.getoption("warmup"):
        PETSc.Sys.Print("WarmUp solve")
        with PETSc.Log.Stage(f"WarmUp solve with {pc_approach}"):
            solver.solve(None, x0)

        # Zero what needs to be zeroed before another solve
        x0.zeroEntries()
        x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        for var in pdeproblem.solution_vars:
            var.vector.zeroEntries()
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # # Yet another sanity check
        # for var in problem.u:
        #     with var.vector.localForm() as loc:
        #         assert not np.all(loc.array_r)

    # Detailed monitoring of convergence history
    # solver.setConvergenceHistory()
    # solver.ksp.setConvergenceHistory()
    if pc_approach == "PCD-AMG":
        pc = solver.ksp.getPC()
        try:
            wrapped_pc = pc.getPythonContext().pc
        except AttributeError:
            pc.setUp()
            wrapped_pc = pc.getPythonContext().pc
        ksp_0, ksp_1 = wrapped_pc.getFieldSplitSubKSP()
        ksp_0.setConvergenceHistory()
        ksp_1.setConvergenceHistory()

    PETSc.Sys.Print("Timed solve")
    with PETSc.Log.Stage(f"Timed solve with {pc_approach}"):
        solver.solve(None, x0)
        info_snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        info_ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        info_pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        info_pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        info_jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        info_residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        time_snes = comm.allreduce(info_snes["time"], op=MPI.SUM) / comm.size
        time_jac = comm.allreduce(info_jac["time"], op=MPI.SUM) / comm.size
        time_residual = comm.allreduce(info_residual["time"], op=MPI.SUM) / comm.size
        time_ksp = comm.allreduce(info_ksp["time"], op=MPI.SUM) / comm.size
        time_pcsetup = comm.allreduce(info_pcsetup["time"], op=MPI.SUM) / comm.size
        time_pcapply = comm.allreduce(info_pcapply["time"], op=MPI.SUM) / comm.size

        # Get outer iterations
        its_snes = solver.getIterationNumber()
        its_ksp = solver.getLinearSolveIterations()

        # Get inner iterations
        # history_snes, its_ksp_detail = solver.getConvergenceHistory()
        # history_ksp = solver.ksp.getConvergenceHistory()
        its_fs0 = 0
        its_fs0_avg = 0
        its_fs1 = 0
        its_fs1_avg = 0
        if pc_approach == "PCD-AMG":
            history_fs0 = ksp_0.getConvergenceHistory()
            history_fs1 = ksp_1.getConvergenceHistory()
            its_fs0 = len(history_fs0) - its_ksp
            its_fs0_avg = its_fs0 / its_ksp
            its_fs1 = len(history_fs1) - its_ksp
            its_fs1_avg = its_fs1 / its_ksp

    pdeproblem.vec_to_functions(x0, pdeproblem.solution_vars)
    s0, s1, s2 = pdeproblem.solution_vars

    # Save results
    module_dir, module_name = os.path.split(os.path.realpath(request.node.fspath))
    results_file = os.path.join(results_dir, f"{os.path.splitext(module_name[5:])[0]}.csv")
    results = {
        "timestamp": timestamp,
        "pc_approach": pc_approach,
        "num_elements": problem.num_elements,
        "num_vertices": problem.num_vertices,
        "num_dofs": problem.num_dofs,
        "num_procs": comm.size,
        "mesh_resolution": problem.N,
        "its_snes": its_snes,
        "its_ksp": its_ksp,
        "its_fs0": its_fs0,
        "its_fs0_avg": its_fs0_avg,
        "its_fs1": its_fs1,
        "its_fs1_avg": its_fs1_avg,
        # "its_ksp_detail": pickle.dumps(its_ksp_detail),
        # "history_snes": pickle.dumps(history_snes),
        # "history_ksp": pickle.dumps(history_ksp),
        "SNESSolve": time_snes,
        "KSPSolve": time_ksp,
        "PCSetUp": time_pcsetup,
        "PCApply": time_pcapply,
        "JacobianEval": time_jac,
        "FunctionEval": time_residual,
    }

    # Save XDMF data
    if not request.config.getoption("noxdmf"):
        pvscript_args = [
            "pvscript_rayleigh_benard.py", os.path.join(results_dir, "fig_rayleigh_benard.png")
        ]
        basename = os.path.splitext(os.path.basename(results_file))[0]
        for field, name in [(s0, "v"), (s1, "p"), (s2, "T")]:
            xfile = f"{basename}_{pc_approach}_nprocs{comm.size}_field_{name}.xdmf"
            xfile = os.path.join(results_dir, xfile)
            with XDMFFile(comm, xfile, "w") as f:
                f.write_mesh(field.function_space.mesh)
                f.write_function(field)
            pvscript_args.append(xfile)

    if comm.rank == 0:
        data = pandas.DataFrame(results, index=[0])
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

    # List timings
    list_timings(comm, [TimingType.wall])

    # Save logs
    logfile = os.path.join(results_dir, f"petsc_rayleigh_benard_{comm.size}.log")
    PETSc.Log.view(viewer=PETSc.Viewer.ASCII(logfile, comm=comm))

    # Clean up options database
    for opt in opts.getAll().keys():
        opts.delValue(opt)


if __name__ == "__main__":
    import argparse
    import sys
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Solve Rayleigh-Benard convection problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sc",
        type=str,
        dest="scaling",
        choices=["weak", "strong"],
        default="weak",
        help="type of scaling to determine global number of DOFs",
    )
    parser.add_argument(
        "--dofs",
        type=float,
        default=30000,
        help="required number of DOFs (per core for weak scaling test)",
    )
    parser.add_argument(
        "--pctype",
        type=str,
        dest="pc_approach",
        default="PCD-AMG",
        choices=["PCD-AMG", "LU"],
        help="Preconditioning approach",
    )
    parser.add_argument("--noxdmf", action="store_true", help="do not save XDMF output")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results file")
    parser.add_argument(
        "--resultsdir",
        metavar="PATH",
        action="store",
        default=None,
        help="directory where to put the results",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="run main solve twice (warm up first)"
    )
    args = parser.parse_args(sys.argv[1:])

    comm = MPI.COMM_WORLD
    if args.scaling == "weak":
        N = target_mesh_size()(comm.size, args.dofs)
    else:
        N = target_mesh_size()(1, args.dofs)
    prob = Problem(comm, N)

    pc_approach = args.pc_approach

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_UTC-%H-%M-%S")

    results_dir = args.resultsdir
    if results_dir is None:
        module_dir = os.path.dirname(os.path.realpath(__file__))  # this file's directory
        results_dir = os.path.join(module_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    class MockedRequest:
        def __init__(self):
            class Config:
                def __init__(self):
                    self.overwrite = args.overwrite
                    self.warmup = args.warmup
                    self.noxdmf = args.noxdmf

                def getoption(self, name):
                    return getattr(self, name)

            class Node:
                def __init__(self):
                    self.fspath = os.path.realpath(__file__)

            self.config = Config()
            self.node = Node()

    request = MockedRequest()

    sys.exit(test_rayleigh_benard(prob, pc_approach, timestamp, results_dir, request))

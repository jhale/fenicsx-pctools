# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
# ---

# # Navier-Stokes equations with a PCD-based preconditioner

# Start with usual imports.

# +
import gmsh
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.mesh import refine
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_vector_block
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags
from dolfiny.function import vec_to_functions
from ufl import inner, grad, div, dot, dx

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block
# -

# Create a distributed (parallel) mesh using the Gmsh API.
# Generate mesh on rank 0, then build a distributed mesh.

# +
gmsh.initialize()

# Set verbosity of Gmsh
gmsh.option.setNumber("General.Terminal", 0)

# Set up the model
model = gmsh.model
model.add("bfstep")

model_rank = 0
mesh_comm = MPI.COMM_WORLD
if mesh_comm.rank == model_rank:
    factory = model.occ
    s1 = factory.addRectangle(-1, 0, 0, 1, 1)
    s2 = factory.addRectangle(0, -1, 0, 5, 2)
    s3 = 3
    factory.fuse([(2, s1)], [(2, s2)], tag=s3)
    factory.synchronize()
    ov = model.getBoundary([(2, s3)])
    l1, l2, l3, l4, l5, l6 = [val[1] for val in ov]  # counterclockwise (l6 = inflow)

    # Tag boundaries
    model.addPhysicalGroup(1, [l6], 1)
    model.setPhysicalName(1, 1, "inflow")
    model.addPhysicalGroup(1, [l4], 2)
    model.setPhysicalName(1, 2, "outflow")
    model.addPhysicalGroup(1, [l1, l2, l3, l5], 3)
    model.setPhysicalName(1, 3, "wall")

    # Set uniform mesh size
    size = 0.2
    all_pts = model.getEntities(0)
    model.mesh.setSize(all_pts, size)

    # Generate mesh
    model.mesh.generate(2)

# Convert Gmsh mesh to DOLFINx
mesh, mts = gmsh_to_dolfin(model, 2, prune_z=True, comm=mesh_comm)
gmsh.finalize()
# -


# TODO: Convert the demo to a plain script, keep it simple!
# NOTE: The `test_` syntax is used for the possibility to attach the debugger from IDE.
def test_run():
    global mesh, mesh_comm

    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    facetdim = tdim - 1

    # Refine mesh to requested level
    level = 0  # FIXME: Make me parameter!
    for _ in range(level):
        mesh = refine(mesh)

    # Build Taylor-Hood function space
    family = "P"
    V_v = fem.VectorFunctionSpace(mesh, (family, 2), dim=gdim)
    V_p = fem.FunctionSpace(mesh, (family, 1))

    v = fem.Function(V_v, name="v")
    v_te = ufl.TestFunction(V_v)

    p = fem.Function(V_p, name="p")
    p_te = ufl.TestFunction(V_p)

    V_char = 1.0
    L_char = 2.0
    Re = 1e1
    nu = fem.Constant(mesh, V_char * L_char / Re)
    F0 = (
        inner(dot(grad(v), v), v_te) * dx
        + nu * inner(grad(v), grad(v_te)) * dx
        - p * div(v_te) * dx
    )
    F1 = -div(v) * p_te * dx

    v_tr, p_tr = ufl.TrialFunction(V_v), ufl.TrialFunction(V_p)
    linearization = "Newton"
    if linearization == "Newton":
        a00 = ufl.derivative(F0, v)
        a01 = ufl.derivative(F0, p)
        a10 = ufl.derivative(F1, v)
        a11 = None
    elif linearization == "Picard":
        a00 = nu * inner(grad(v_tr), grad(v_te)) * dx + inner(dot(grad(v_tr), v), v_te) * dx
        a01 = -p_tr * div(v_te) * dx
        a10 = -div(v_tr) * p_te * dx
        a11 = None
    else:
        raise NotImplementedError(f"Unknown type of linearization '{linearization}'")

    pc_approach = "ILU"
    if pc_approach == "ILU":
        a11 = fem.Function(V_p, name="zero_p") * p_tr * p_te * dx

    def v_inflow_eval(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 4.0 * x[1] * (1.0 - x[1])
        values[1] = np.zeros(x.shape[1])
        return values

    # FIXME: Move this up to gmsh script!
    tag_inflow = 1

    v_inflow = fem.Function(V_v)
    v_inflow.interpolate(v_inflow_eval)
    inflowdofsV_v = fem.locate_dofs_topological(V_v, facetdim, mts["inflow"].indices)
    inflowdofsV_p = fem.locate_dofs_topological(V_p, facetdim, mts["inflow"].indices)

    outflowdofsV_p = fem.locate_dofs_topological(V_p, facetdim, mts["outflow"].indices)

    v_walls = fem.Function(V_v)
    wallsdofsV_v = fem.locate_dofs_topological(V_v, facetdim, mts["wall"].indices)
    wallsdofsV_p = fem.locate_dofs_topological(V_p, facetdim, mts["wall"].indices)

    # Find common dofs at corners and remove them for consistency
    incornerdofsV_v = np.intersect1d(inflowdofsV_v, wallsdofsV_v)
    inflowdofsV_v = np.setdiff1d(inflowdofsV_v, incornerdofsV_v)
    incornerdofsV_p = np.intersect1d(inflowdofsV_p, wallsdofsV_p)
    inflowdofsV_p = np.setdiff1d(inflowdofsV_p, incornerdofsV_p)
    outcornerdofsV_p = np.intersect1d(outflowdofsV_p, wallsdofsV_p)
    outflowdofsV_p = np.setdiff1d(outflowdofsV_p, outcornerdofsV_p)

    bcs = [fem.dirichletbc(v_inflow, inflowdofsV_v), fem.dirichletbc(v_walls, wallsdofsV_v)]
    bcs_pcd = []
    if pc_approach == "PCDv1":
        bcs_pcd.append(fem.dirichletbc(fem.Function(V_p), inflowdofsV_p))
    elif pc_approach == "PCDv2":
        bcs_pcd.append(fem.dirichletbc(fem.Function(V_p), outflowdofsV_p))

    # Set up PETSc options
    PETSc.Log.begin()
    opts = PETSc.Options()
    opts["options_left"] = None

    problem_prefix = "ns_"
    opts.prefixPush(problem_prefix)
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-05
    opts["snes_max_it"] = 25

    # -- linear solver configuration
    opts["ksp_converged_reason"] = None
    # opts["ksp_monitor_true_residual"] = None
    opts["ksp_rtol"] = 1e-06
    opts["ksp_max_it"] = 1000
    if pc_approach == "LU":
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts["wrapped_pc_type"] = "lu"
        opts["wrapped_pc_factor_mat_solver_type"] = "mumps"
    if pc_approach == "ILU":
        opts["ksp_type"] = "gmres"
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts["wrapped_pc_type"] = "hypre"
        opts["wrapped_pc_hypre_type"] = "euclid"
    elif pc_approach.startswith("PCD"):
        pcd_type = {"PCDv1": "fenicsx_pctools.la.PCDPC_BRM1", "PCDv2": "fenicsx_pctools.la.PCDPC_BRM2"}[pc_approach]
        opts["ksp_type"] = "gmres"
        opts["ksp_gmres_restart"] = 150
        opts["ksp_pc_side"] = "right"
        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "fieldsplit"
        opts["pc_fieldsplit_type"] = "schur"
        opts["pc_fieldsplit_schur_fact_type"] = "upper"
        opts["pc_fieldsplit_schur_precondition"] = "user"
        opts["pc_fieldsplit_0_fields"] = 0
        opts["pc_fieldsplit_1_fields"] = 1
        opts["fieldsplit_0_ksp_type"] = "preonly"
        opts["fieldsplit_0_pc_type"] = "python"
        opts["fieldsplit_0_pc_python_type"] = "fenicsx_pctools.WrappedPC"
        opts["fieldsplit_0_wrapped_pc_type"] = "lu"
        opts["fieldsplit_1_ksp_type"] = "preonly"
        opts["fieldsplit_1_pc_type"] = "python"
        opts["fieldsplit_1_pc_python_type"] = pcd_type
        opts["fieldsplit_1_pcd_Mp_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Mp_pc_type"] = "lu"
        opts["fieldsplit_1_pcd_Ap_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Ap_pc_type"] = "lu"
        opts.prefixPop()  #: wrapped_
    opts.prefixPop()  #: ns_

    # Set up solver
    class SNESContext(object):
        def __init__(self, u, bcs, F_form, J_form, appctx):
            self.u = u
            self.bcs = bcs
            self.F = create_vector_block(F_form)
            self.F_form = F_form
            self.J = create_splittable_matrix_block(
                J_form, bcs, appctx, options_prefix=problem_prefix
            )
            self.J_form = J_form
            self.x = self.F.copy()
            self.solution = [fem.Function(u_i.function_space, name=u_i.name) for u_i in u]

        def update_functions(self, x):
            vec_to_functions(x, self.u)
            x.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        def eval_residual(self, snes, x, F):
            with self.F.localForm() as f_local:
                f_local.set(0.0)
            self.update_functions(x)

            assemble_vector_block(
                self.F, self.F_form, self.J_form, self.bcs, x0=self.x, scale=-1.0
            )

        def eval_jacobian(self, snes, x, J, P):
            self.J.zeroEntries()
            self.update_functions(x)

            if self.J.getType() != "python":
                assemble_matrix_block(self.J, self.J_form, self.bcs, diagonal=1.0)
            self.J.assemble()

    F_form = [F0, F1]
    J_form = [[a00, a01], [a10, a11]]

    regions = merge_meshtags(mts, facetdim)
    appctx = {
        "nu": nu,
        "v": v,
        "bcs_pcd": bcs_pcd,
        "ds_in": ufl.Measure("ds", domain=mesh, subdomain_data=regions, subdomain_id=tag_inflow),
    }
    snesctx = SNESContext([v, p], bcs, fem.form(F_form), fem.form(J_form), appctx)

    solver = PETSc.SNES().create(mesh_comm)
    solver.setFunction(snesctx.eval_residual, snesctx.F)
    solver.setJacobian(snesctx.eval_jacobian, snesctx.J)
    solver.setOptionsPrefix(problem_prefix)
    solver.setFromOptions()

    # if request.config.getoption("warmup"):
    #     PETSc.Sys.Print("WarmUp solve")
    #     with PETSc.Log.Stage(f"WarmUp solve with {pc_approach} ({linearization}), Re = {Re}"):
    #         solver.solve(None, snesctx.x)
    #     snesctx.x.zeroEntries()

    # Solve the problem
    PETSc.Sys.Print("Timed solve")
    solver.setConvergenceHistory()
    solver.ksp.setConvergenceHistory()
    with PETSc.Log.Stage(f"Timed solve with {pc_approach} ({linearization}), Re = {Re}"):
        solver.solve(None, snesctx.x)
        info_snes = PETSc.Log.Event("SNESSolve").getPerfInfo()
        info_ksp = PETSc.Log.Event("KSPSolve").getPerfInfo()
        info_pcsetup = PETSc.Log.Event("PCSetUp").getPerfInfo()
        info_pcapply = PETSc.Log.Event("PCApply").getPerfInfo()
        info_jac = PETSc.Log.Event("SNESJacobianEval").getPerfInfo()
        info_residual = PETSc.Log.Event("SNESFunctionEval").getPerfInfo()
        time_snes = mesh_comm.allreduce(info_snes["time"], op=MPI.SUM) / mesh_comm.size
        time_jac = mesh_comm.allreduce(info_jac["time"], op=MPI.SUM) / mesh_comm.size
        time_residual = mesh_comm.allreduce(info_residual["time"], op=MPI.SUM) / mesh_comm.size
        time_ksp = mesh_comm.allreduce(info_ksp["time"], op=MPI.SUM) / mesh_comm.size
        time_pcsetup = mesh_comm.allreduce(info_pcsetup["time"], op=MPI.SUM) / mesh_comm.size
        time_pcapply = mesh_comm.allreduce(info_pcapply["time"], op=MPI.SUM) / mesh_comm.size

        its_snes = solver.getIterationNumber()
        its_ksp = solver.getLinearSolveIterations()

        # FIXME: We are not interested in the following info at the moment. Remove these lines!
        # history_snes, its_ksp_detail = solver.getConvergenceHistory()
        # history_ksp = solver.ksp.getConvergenceHistory()

    vec_to_functions(snesctx.x, snesctx.solution)
    s0, s1 = snesctx.solution

    # Save ParaView plots
    for field, name in [(s0, "v"), (s1, "p")]:
        xfile = f"solution_{name}.xdmf"
        with XDMFFile(mesh_comm, xfile, "w") as f:
            f.write_mesh(field.function_space.mesh)
            f.write_function(field)

    # Clean up options database
    for opt in opts.getAll().keys():
        opts.delValue(opt)


if __name__ == "__main__":
    test_run()

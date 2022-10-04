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

# TODO: Add problem description with references!

# Start with usual imports.

# +
import gmsh
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block, create_vector_block
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags
from dolfiny.function import vec_to_functions
from ufl import inner, grad, div, dot, dx

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block
# -


# Create a distributed (parallel) mesh using the Gmsh API and mark exterior facets.
# The mesh is generated on rank 0 and then distrubuted across other ranks.


# +
def get_mesh_data(refinement_level=0, mesh_comm=MPI.COMM_WORLD):
    gmsh.initialize()

    # Set verbosity of Gmsh
    gmsh.option.setNumber("General.Terminal", 0)

    # Set up the model
    model = gmsh.model
    model.add("bfstep")

    model_rank = 0
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
        model.setPhysicalName(1, 1, "inlet")
        model.addPhysicalGroup(1, [l4], 2)
        model.setPhysicalName(1, 2, "outlet")
        model.addPhysicalGroup(1, [l1, l2, l3, l5], 3)
        model.setPhysicalName(1, 3, "wall")

        # Set uniform mesh size at all points
        size = (0.5 ** refinement_level) * 0.2
        model.mesh.setSize(model.getEntities(0), size)

        # Generate 2D mesh
        model.mesh.generate(2)

    # Convert Gmsh mesh to DOLFINx
    mesh, mesh_tags = gmsh_to_dolfin(model, 2, prune_z=True, comm=mesh_comm)
    gmsh.finalize()

    return mesh, mesh_tags
# -

# All other steps are kept under the function ``test_navier_stokes``, so it is easy to run
# the demo repeatedly with different preconditioning and linearization schemas (see below).


# +
# FIXME: Run this with PCD!
def test_navier_stokes(mesh_data=None, pc_approach="ILU", linearization="Newton"):
    mesh, mts = mesh_data or get_mesh_data()
    mesh_comm = mesh.comm
# -

# Build function spaces, define non-linear variational forms representing the Navier-Stokes,
# and calculate the matrix blocks for the linearized problem.

# +
    # Build Taylor-Hood function space
    V_v = fem.VectorFunctionSpace(mesh, ("P", 2), dim=mesh.geometry.dim)
    V_p = fem.FunctionSpace(mesh, ("P", 1))

    v = fem.Function(V_v, name="v")
    p = fem.Function(V_p, name="p")
    v_te = ufl.TestFunction(V_v)
    p_te = ufl.TestFunction(V_p)

    Re = 10  # Reynolds number
    V_char = 1.0  # characteristic length
    L_char = 2.0  # characteristic velocity
    nu = fem.Constant(mesh, V_char * L_char / Re)  # kinematic viscosity
    F0 = (
        inner(dot(grad(v), v), v_te) * dx
        + nu * inner(grad(v), grad(v_te)) * dx
        - p * div(v_te) * dx
    )
    F1 = -div(v) * p_te * dx

    v_tr, p_tr = ufl.TrialFunction(V_v), ufl.TrialFunction(V_p)
    if linearization == "Newton":
        a00 = ufl.derivative(F0, v)
        a01 = ufl.derivative(F0, p)
        a10 = ufl.derivative(F1, v)
    elif linearization == "Picard":
        a00 = nu * inner(grad(v_tr), grad(v_te)) * dx + inner(dot(grad(v_tr), v), v_te) * dx
        a01 = -p_tr * div(v_te) * dx
        a10 = -div(v_tr) * p_te * dx
    else:
        raise NotImplementedError(f"Unknown type of linearization '{linearization}'")
    a11 = fem.Function(V_p, name="zero_p") * p_tr * p_te * dx if pc_approach == "ILU" else None

    F_form = [F0, F1]
    J_form = [[a00, a01], [a10, a11]]
# -

# Define primary and secondary boundary conditions.
# The latter are important for the performance of PCD preconditioners.

# +
    def v_inflow_eval(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = 4.0 * x[1] * (1.0 - x[1])
        values[1] = np.zeros(x.shape[1])
        return values

    v_wall = fem.Function(V_v)
    v_inflow = fem.Function(V_v)
    v_inflow.interpolate(v_inflow_eval)

    # Localize boundary dofs
    fdim = mesh.topology.dim - 1
    wall_dofs_v = fem.locate_dofs_topological(V_v, fdim, mts["wall"].indices)
    wall_dofs_p = fem.locate_dofs_topological(V_p, fdim, mts["wall"].indices)
    inlet_dofs_v = fem.locate_dofs_topological(V_v, fdim, mts["inlet"].indices)
    inlet_dofs_p = fem.locate_dofs_topological(V_p, fdim, mts["inlet"].indices)
    outlet_dofs_p = fem.locate_dofs_topological(V_p, fdim, mts["outlet"].indices)

    # Find common dofs at corners and remove them for consistency
    inlet_cdofs_v = np.intersect1d(inlet_dofs_v, wall_dofs_v)
    inlet_dofs_v = np.setdiff1d(inlet_dofs_v, inlet_cdofs_v)
    inlet_cdofs_p = np.intersect1d(inlet_dofs_p, wall_dofs_p)
    inlet_dofs_p = np.setdiff1d(inlet_dofs_p, inlet_cdofs_p)
    outlet_cdofs_p = np.intersect1d(outlet_dofs_p, wall_dofs_p)
    outlet_dofs_p = np.setdiff1d(outlet_dofs_p, outlet_cdofs_p)

    # Collect primary boundary condtions
    bcs = [fem.dirichletbc(v_inflow, inlet_dofs_v), fem.dirichletbc(v_wall, wall_dofs_v)]

    # Collect secondary boundary condtions (only active with PCD-based preconditioning)
    try:
        bcs_pcd = {
            "PCDPC_vX": [fem.dirichletbc(fem.Function(V_p), inlet_dofs_p)],
            "PCDPC_vY": [fem.dirichletbc(fem.Function(V_p), outlet_dofs_p)],
        }[pc_approach]
    except KeyError:
        bcs_pcd = []
# -

# Define the context for the nonlinear SNES solver that will be used to solve the problem.

# TODO: Comment on the use of _splittable_ matrix when evaluating the Jacobian!

# +
    # Prepare solver context
    class SNESContext(object):
        def __init__(self, u, bcs, F_form, J_form, appctx, options_prefix):
            self.u = u
            self.bcs = bcs
            self.F = create_vector_block(F_form)
            self.F_form = F_form
            self.J = create_splittable_matrix_block(
                J_form, bcs, appctx, options_prefix=options_prefix
            )
            self.J_form = J_form
            self.x = self.F.copy()
            self.solution = [fem.Function(u_i.function_space, name=u_i.name) for u_i in u]
            self.options_prefix = options_prefix

        def update_functions(self, x):
            vec_to_functions(x, self.u)
            x.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        def eval_residual(self, snes, x, F):
            with self.F.localForm() as f_local:
                f_local.set(0.0)
            self.update_functions(x)
            assemble_vector_block(self.F, self.F_form, self.J_form, self.bcs, x0=self.x, scale=-1.0)

        def eval_jacobian(self, snes, x, J, P):
            self.J.zeroEntries()
            self.update_functions(x)
            if self.J.getType() != "python":
                assemble_matrix_block(self.J, self.J_form, self.bcs, diagonal=1.0)
            self.J.assemble()

    regions = merge_meshtags(mts, fdim)
    tag_inlet = np.unique(mts["inlet"].values)[0]
    ds_in = ufl.Measure("ds", domain=mesh, subdomain_data=regions, subdomain_id=tag_inlet)

    appctx = {"nu": nu, "v": v, "bcs_pcd": bcs_pcd, "ds_in": ds_in}
    snesctx = SNESContext([v, p], bcs, fem.form(F_form), fem.form(J_form), appctx, "ns_")
# -

# Configure the solver depending on the chosen preconditioning approach.

# TODO: Comment on the use of ``WrappedPC``!

# +
    # Set up PETSc options
    PETSc.Log.begin()
    opts = PETSc.Options()
    opts["options_left"] = None

    opts.prefixPush(snesctx.options_prefix)
    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-05
    opts["snes_max_it"] = 25

    # -- linear solver configuration
    opts["ksp_converged_reason"] = None
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
        opts["fieldsplit_1_pc_python_type"] = f"fenicsx_pctools.{pc_approach}"
        opts["fieldsplit_1_pcd_Mp_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Mp_pc_type"] = "lu"
        opts["fieldsplit_1_pcd_Ap_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Ap_pc_type"] = "lu"
        opts.prefixPop()  # wrapped_
    else:
        raise NotImplementedError(f"Unknown type of preconditioner '{pc_approach}'")
    opts.prefixPop()  # ns_

    # Set up nonlinear solver
    solver = PETSc.SNES().create(mesh_comm)
    solver.setFunction(snesctx.eval_residual, snesctx.F)
    solver.setJacobian(snesctx.eval_jacobian, snesctx.J)
    solver.setOptionsPrefix(snesctx.options_prefix)
    solver.setFromOptions()
# -

# Now we are ready to solve the problem and track the number of iterations.

# +
    # Solve the problem
    PETSc.Sys.Print("Solving the nonlinear problem using SNES")
    solver.setConvergenceHistory()
    solver.ksp.setConvergenceHistory()
    with PETSc.Log.Stage(f"SNES solve with {pc_approach} ({linearization}), Re = {Re}"):
        solver.solve(None, snesctx.x)
        its_snes = solver.getIterationNumber()
        its_ksp = solver.getLinearSolveIterations()
    PETSc.Sys.Print(
        f"Solver converged in {its_snes} nonlinear iterations"
        f" (with total number of {its_ksp} linear iterations)"
    )

    vec_to_functions(snesctx.x, snesctx.solution)
    s0, s1 = snesctx.solution

    results = {
        "fields_names": [(s0, "v"), (s1, "p")],
        "its_snes": its_snes,
        "its_ksp": its_ksp,
    }

    # Clean up options database
    for opt in opts.getAll().keys():
        opts.delValue(opt)

    return results
# -

# Users are encouraged to experiment with different preconditioning and linearization schemas.

# TODO: Possibly write a short loop taking into account mesh refinements and/or different Reynolds
# numbers to demonstrate robustness of PCD preconditioner.
#
# TODO: We may want to run this demo as part of CI pipeline.


# +
if __name__ == "__main__":
    mesh_comm = MPI.COMM_WORLD
    mesh_data = get_mesh_data(refinement_level=0, mesh_comm=mesh_comm)
    results = test_navier_stokes(mesh_data, pc_approach="ILU", linearization="Newton")

    # Save ParaView plots
    for field, name in results["fields_names"]:
        xfile = f"solution_{name}.xdmf"
        with XDMFFile(mesh_comm, xfile, "w") as f:
            f.write_mesh(field.function_space.mesh)
            f.write_function(field)
# -

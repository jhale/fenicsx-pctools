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

# TODO: We may want to run this demo as part of CI pipeline.

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
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags
from dolfiny.function import vec_to_functions
from ufl import inner, grad, div, dot, dx

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block

# -

# Create a distributed (parallel) mesh using the Gmsh API and mark exterior facets.
# The mesh is generated on rank 0 and then distrubuted across other ranks.


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
    model.setPhysicalName(1, 1, "inlet")
    model.addPhysicalGroup(1, [l4], 2)
    model.setPhysicalName(1, 2, "outlet")
    model.addPhysicalGroup(1, [l1, l2, l3, l5], 3)
    model.setPhysicalName(1, 3, "wall")

    # Set uniform mesh size at all points
    mesh_size = 0.1
    model.mesh.setSize(model.getEntities(0), mesh_size)

    # Generate 2D mesh
    model.mesh.generate(2)

# Convert Gmsh mesh to DOLFINx
mesh, mts = gmsh_to_dolfin(model, 2, prune_z=True, comm=mesh_comm)
gmsh.finalize()
# -

# Build function spaces, define non-linear variational forms representing the incompressible
# Navier-Stokes equations, and calculate the matrix blocks for the linearized problem.


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
F0 = inner(dot(grad(v), v), v_te) * dx + nu * inner(grad(v), grad(v_te)) * dx - p * div(v_te) * dx
F1 = -div(v) * p_te * dx

a00 = ufl.derivative(F0, v)
a01 = ufl.derivative(F0, p)
a10 = ufl.derivative(F1, v)
a11 = None

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

# Collect secondary boundary condtions for PCD preconditioners
pcd_type = "PCDPC_vY"  # pick one of the two versions specified in the dictionary below
bcs_pcd = {
    "PCDPC_vX": [fem.dirichletbc(fem.Function(V_p), inlet_dofs_p)],
    "PCDPC_vY": [fem.dirichletbc(fem.Function(V_p), outlet_dofs_p)],
}[pcd_type]
# -

# Define the context for the nonlinear SNES solver that will be used to solve the problem.

# TODO: Comment on the use of _splittable_ matrix when evaluating the Jacobian!


# +
# Wrap PDEs, BCs and solution variables into a class that can assemble Jacobian and residual
class PDEProblem:
    def __init__(self, F_form, J_form, solution_vars, bcs, P_form=None):
        self.F_form = F_form
        self.J_form = J_form
        self.P_form = P_form
        self.bcs = bcs
        self.solution_vars = solution_vars

    def F_block(self, snes, x, F):
        with F.localForm() as f_local:
            f_local.set(0.0)  # NOTE: f_local includes ghosts

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        vec_to_functions(x, self.solution_vars)

        fem.petsc.assemble_vector_block(F, self.F_form, self.J_form, self.bcs, x0=x, scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def J_block(self, snes, x, J, P):
        J.zeroEntries()
        if J.getType() != "python":
            fem.petsc.assemble_matrix_block(J, self.J_form, self.bcs, diagonal=1.0)
        J.assemble()
        if self.P_form is not None:
            P.zeroEntries()
            if P.getType() != "python":
                fem.petsc.assemble_matrix_block(P, self.P_form, self.bcs, diagonal=1.0)
            P.assemble()


problem_prefix = "ns_"
regions, tag_id_map = merge_meshtags(mts, fdim)
ds_in = ufl.Measure("ds", domain=mesh, subdomain_data=regions, subdomain_id=tag_id_map["inlet"])
appctx = {"nu": nu, "v": v, "bcs_pcd": bcs_pcd, "ds_in": ds_in}

# Prepare Jacobian matrix (UFL form is required in this step)
Jmat = create_splittable_matrix_block(
    J_form, bcs, appctx, comm=mesh_comm, options_prefix=problem_prefix
)

F_form = fem.form(F_form)
J_form = fem.form(J_form)
pdeproblem = PDEProblem(F_form, J_form, [v, p], bcs)

# Prepare vectors (DOLFINx form is required here)
Fvec = fem.petsc.create_vector_block(F_form)
x0 = fem.petsc.create_vector_block(F_form)
# -

# Configure the solver depending on the chosen preconditioning approach.

# TODO: Comment on the use of ``WrappedPC``!


# +
# Set up PETSc options
PETSc.Log.begin()
opts = PETSc.Options()
opts.prefixPush(problem_prefix)
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-05
opts["snes_max_it"] = 25
opts["ksp_converged_reason"] = None
opts["ksp_rtol"] = 1e-06
opts["ksp_max_it"] = 1000
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
opts["fieldsplit_1_pc_python_type"] = f"fenicsx_pctools.{pcd_type}"
opts["fieldsplit_1_pcd_Mp_ksp_type"] = "preonly"
opts["fieldsplit_1_pcd_Mp_pc_type"] = "lu"
opts["fieldsplit_1_pcd_Ap_ksp_type"] = "preonly"
opts["fieldsplit_1_pcd_Ap_pc_type"] = "lu"
opts.prefixPop()  # wrapped_
opts.prefixPop()  # ns_

# Set up nonlinear solver
solver = PETSc.SNES().create(mesh_comm)
solver.setFunction(pdeproblem.F_block, Fvec)
solver.setJacobian(pdeproblem.J_block, J=Jmat, P=None)
solver.setOptionsPrefix(problem_prefix)
solver.setFromOptions()
# -

# Now we are ready to solve the problem and track the number of iterations.


# +
# Solve the problem
PETSc.Sys.Print("Solving the nonlinear problem with SNES")
solver.solve(None, x0)
its_snes = solver.getIterationNumber()
its_ksp = solver.getLinearSolveIterations()
PETSc.Sys.Print(
    f"Solver converged in {its_snes} nonlinear iterations"
    f" (with total number of {its_ksp} linear iterations)"
)

# Update solution variables
vec_to_functions(x0, pdeproblem.solution_vars)
# -

# Visualize the resulting fields and do the cleanup.


# +
# Save ParaView plots
for field in pdeproblem.solution_vars:
    xfile = f"solution_{field.name}.xdmf"
    with XDMFFile(mesh_comm, xfile, "w") as f:
        f.write_mesh(field.function_space.mesh)
        f.write_function(field)

# Destroy PETSc objects
solver.destroy()
Jmat.destroy()
# -

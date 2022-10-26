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

import gmsh
import numpy as np
import pathlib

from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfiny.mesh import gmsh_to_dolfin, merge_meshtags
from dolfiny.function import vec_to_functions
from ufl import inner, grad, div, dot, dx

from fenicsx_pctools.mat.splittable import create_splittable_matrix_block


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

F_ufl = [F0, F1]
J_ufl = [[a00, a01], [a10, a11]]
F_dfx = fem.form(F_ufl)
J_dfx = fem.form(J_ufl)


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
inlet_dofs_v = fem.locate_dofs_topological(V_v, fdim, mts["inlet"].indices)
inlet_dofs_p = fem.locate_dofs_topological(V_p, fdim, mts["inlet"].indices)
outlet_dofs_p = fem.locate_dofs_topological(V_p, fdim, mts["outlet"].indices)

# Collect primary boundary condtions
bcs = [fem.dirichletbc(v_inflow, inlet_dofs_v), fem.dirichletbc(v_wall, wall_dofs_v)]

# Collect secondary boundary condtions for PCD preconditioner
pcd_type = "PCDPC_vY"  # pick one of the two versions specified in the dictionary below
bcs_pcd = {
    "PCDPC_vX": [fem.dirichletbc(fem.Function(V_p), inlet_dofs_p)],
    "PCDPC_vY": [fem.dirichletbc(fem.Function(V_p), outlet_dofs_p)],
}[pcd_type]


# Wrap PDEs, BCs and solution variables into a class that can assemble Jacobian and residual
class PDEProblem:
    def __init__(self, F_dfx, J_dfx, solution_vars, bcs, P_dfx=None):
        self.F_dfx = F_dfx
        self.J_dfx = J_dfx
        self.P_dfx = P_dfx
        self.bcs = bcs
        self.solution_vars = solution_vars

    def F_block(self, snes, x, F):
        with F.localForm() as f_local:
            f_local.set(0.0)  # NOTE: f_local includes ghosts

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        vec_to_functions(x, self.solution_vars)

        fem.petsc.assemble_vector_block(F, self.F_dfx, self.J_dfx, self.bcs, x0=x, scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def J_block(self, snes, x, J, P):
        J_mat = J.getPythonContext().Mat if J.getType() == "python" else J
        J.zeroEntries()
        fem.petsc.assemble_matrix_block(J_mat, self.J_dfx, self.bcs, diagonal=1.0)
        J.assemble()
        if self.P_dfx is not None:
            P_mat = P.getPythonContext().Mat if P.getType() == "python" else P
            P.zeroEntries()
            fem.petsc.assemble_matrix_block(P_mat, self.P_dfx, self.bcs, diagonal=1.0)
            P.assemble()


problem_prefix = "ns_"
regions, tag_id_map = merge_meshtags(mts, fdim)
ds_in = ufl.Measure("ds", domain=mesh, subdomain_data=regions, subdomain_id=tag_id_map["inlet"])
appctx = {"nu": nu, "v": v, "bcs_pcd": bcs_pcd, "ds_in": ds_in}

# Prepare Jacobian matrix and its splittable version
J_mat = fem.petsc.assemble_matrix_block(J_dfx, bcs)
J_mat.assemble()

J_splittable = create_splittable_matrix_block(J_mat, J_ufl, **appctx)
J_splittable.setOptionsPrefix(problem_prefix)

pdeproblem = PDEProblem(F_dfx, J_dfx, [v, p], bcs)

# Prepare vectors (DOLFINx form is required here)
F_vec = fem.petsc.create_vector_block(F_dfx)
x0 = fem.petsc.create_vector_block(F_dfx)

# Set up PETSc options
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
solver.setFunction(pdeproblem.F_block, F_vec)
solver.setJacobian(pdeproblem.J_block, J=J_splittable, P=None)
solver.setOptionsPrefix(problem_prefix)
solver.setFromOptions()

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

# Save ParaView plots
outdir = pathlib.Path(__file__).resolve().parent.joinpath("output")
for field in pdeproblem.solution_vars:
    xfile = outdir.joinpath(f"solution_{field.name}.xdmf")
    with XDMFFile(mesh_comm, xfile, "w") as f:
        f.write_mesh(field.function_space.mesh)
        f.write_function(field)

# Destroy PETSc objects
solver.destroy()
J_splittable.destroy()
J_mat.destroy()

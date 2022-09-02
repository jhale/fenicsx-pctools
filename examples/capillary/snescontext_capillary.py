import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem


class SNESContext:
    def __init__(self, F, J, solution_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.solution_vars = solution_vars

        self.norm_r = {}
        self.norm_dx = {}
        self.norm_x = {}

        self.comm = self.solution_vars[0].function_space.mesh.comm

    @staticmethod
    def vec_to_functions(x, u):
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] = x.array_r[offset : offset + size_local]
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
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

        fem.assemble_vector_block(F, self.L, self.a, self.bcs, x0=x, scale=-1.0)
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

    def compute_norms(self, snes):
        comm = self.comm

        r = snes.getFunction()[0].getArray(readonly=True)
        dx = snes.getSolutionUpdate().getArray(readonly=True)
        x = snes.getSolution().getArray(readonly=True)

        ei_r = []
        ei_dx = []
        ei_x = []

        offset = 0
        for i, ui in enumerate(self.solution_vars):
            size_local = ui.vector.getLocalSize()

            subvec_r = r[offset : offset + size_local]
            subvec_dx = dx[offset : offset + size_local]
            subvec_x = x[offset : offset + size_local]

            # Need first apply square, only then sum over processes
            # i.e. norm is not a linear function
            ei_r.append(np.sqrt(comm.allreduce(np.linalg.norm(subvec_r) ** 2, op=MPI.SUM)))
            ei_dx.append(np.sqrt(comm.allreduce(np.linalg.norm(subvec_dx) ** 2, op=MPI.SUM)))
            ei_x.append(np.sqrt(comm.allreduce(np.linalg.norm(subvec_x) ** 2, op=MPI.SUM)))

            offset += size_local

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x

    def print_norms(self, it, tab):
        for i, ui in enumerate(self.solution_vars):
            PETSc.Sys.Print(
                tab
                + "# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                    i, self.norm_x[it][i], self.norm_dx[it][i], self.norm_r[it][i], ui.name
                )
            )
        PETSc.Sys.Print(
            tab
            + "# all    |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(
                np.linalg.norm(np.asarray(self.norm_x[it])),
                np.linalg.norm(np.asarray(self.norm_dx[it])),
                np.linalg.norm(np.asarray(self.norm_r[it])),
            )
        )

    def monitor(self, snes, it, norm):
        tab = (snes.getTabLevel() + 2) * " "
        PETSc.Sys.Print(f"\n{tab}### SNES iteration {it}")
        self.compute_norms(snes)
        self.print_norms(it, tab)

    def reset(self):
        self.norm_r.clear()
        self.norm_dx.clear()
        self.norm_x.clear()

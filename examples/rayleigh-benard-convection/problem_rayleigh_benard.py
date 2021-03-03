import numpy as np
import ufl

from functools import cached_property
from collections import OrderedDict
from ufl import dot, inner, grad, div, dx
from dolfinx import cpp, fem, UnitCubeMesh, Constant, Function, FunctionSpace, VectorFunctionSpace
from dolfinx.mesh import MeshTags, locate_entities


# FIXME: This is too general, make it simple for this specific example!
def get_boundary_mask(mesh, dim, ghosted=True):
    r"""Get a mask for entities of given (co)dimension indicating whether these entities lay
    on the boundary of the given mesh. The length of the returned array is by default equal
    to the number of entities owned by the given process plus the number of ghost entities.

    Parameters:
        mesh (dfx.Mesh): mesh containing the entities
        cdim (int): (co)dimension (default: -1)
        ghosted (bool): extend the mask by ghost entities (default: True)

    Returns:
        np.ndarray: array of boolean values indexed by local entity indices
    """
    from dolfinx.mesh import locate_entities_boundary
    from petsc4py import PETSc

    topology = mesh.topology
    tdim = topology.dim

    # Create required connectivities
    topology.create_connectivity(tdim - 1, tdim)  # mark interior/exterior facets
    if dim == 1 and tdim > 2:
        topology.create_connectivity(1, tdim - 1)
    # topology.create_connectivity_all()  # NOTE: Use this if the above shows to be insufficient.

    # Get boundary mask (not yet updated)
    imap = topology.index_map(dim)
    if dim < tdim:
        bndry_entities = locate_entities_boundary(mesh, dim, lambda x: np.full(x.shape[1], True))
    else:  # dim == tdim
        bndry_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
        c = topology.connectivity(tdim - 1, tdim)
        bndry_entities = np.unique([c.links(facet_id)[0] for facet_id in bndry_facets])
    bndry_mask = np.full(imap.size_local + imap.num_ghosts, False)
    bndry_mask[bndry_entities] = True
    # NOTE (from the documentation of `dolfinx.mesh.locate_entities_boundary`):
    #   For vertices and edges, in parallel this function will not necessarily
    #   mark all entities that are on the exterior boundary. For example, it is
    #   possible for a process to have a vertex that lies on the boundary without
    #   any of the attached facets being a boundary facet.

    # Update values in boundary mask from ghosts
    bs = 1  # TODO: Double check that the block size is always 1 for topology index maps.
    x = cpp.la.create_vector(imap, bs)
    indices = np.arange(0, imap.size_local, dtype=np.intc)
    x.setValuesLocal(indices, bndry_mask[: imap.size_local], addv=PETSc.InsertMode.INSERT)
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    with x.localForm() as xloc:
        bndry_mask = np.array(xloc.getArray(), dtype=bool)

    if not ghosted:
        bndry_mask = bndry_mask[: imap.size_local]

    return bndry_mask


class Problem(object):
    def __init__(self, comm, N=10):
        self.comm = comm
        self.N = N

        self.coeffs = dict(Ra=200.0, Pr=6.8, g=(0.0, 0.0, -1.0))

    @cached_property
    def domain_data(self):
        cell_type = cpp.mesh.CellType.tetrahedron
        # cell_type = cpp.mesh.CellType.hexahedron
        ghost_mode = cpp.mesh.GhostMode.shared_facet

        mesh = UnitCubeMesh(self.comm, self.N, self.N, self.N, cell_type, ghost_mode)
        mesh.topology.create_connectivity_all()
        imap = mesh.topology.index_map(2)
        indices = np.arange(0, imap.size_local + imap.num_ghosts)
        regions = MeshTags(mesh, 2, indices, np.zeros_like(indices, dtype=np.intc))
        map_target_idx = {"left": 1, "right": 2, "rest": 3}

        bndry_mask = get_boundary_mask(mesh, 2)
        regions.values[bndry_mask] = map_target_idx["rest"]

        left_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 0.0))
        regions.values[left_facets] = map_target_idx["left"]

        right_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 1.0))
        regions.values[right_facets] = map_target_idx["right"]

        return mesh, regions, map_target_idx

    @cached_property
    def num_elements(self):
        mesh = self.domain_data[0]
        tdim = mesh.topology.dim
        mesh.topology.create_entities(tdim)
        return mesh.topology.index_map(tdim).size_global

    @cached_property
    def num_vertices(self):
        mesh = self.domain_data[0]
        return mesh.topology.index_map(0).size_global

    @cached_property
    def Ra(self):
        mesh = self.domain_data[0]
        return Constant(mesh, self.coeffs["Ra"])

    @cached_property
    def Pr(self):
        mesh = self.domain_data[0]
        return Constant(mesh, self.coeffs["Pr"])

    @cached_property
    def g(self):
        mesh = self.domain_data[0]
        return Constant(mesh, self.coeffs["g"])

    @cached_property
    def _mixed_space(self):
        mesh = self.domain_data[0]
        family = "P" if mesh.ufl_cell() == ufl.tetrahedron else "Q"

        return OrderedDict(
            v=VectorFunctionSpace(mesh, (family, 2), dim=3),
            p=FunctionSpace(mesh, (family, 1)),
            T=FunctionSpace(mesh, (family, 1)),
        )

    @property
    def function_names(self):
        return tuple(self._mixed_space.keys())

    @property
    def function_spaces(self):
        return tuple(self._mixed_space.values())

    @property
    def num_dofs(self):
        num_dofs = 0
        for f in self.u:
            num_dofs += f.vector.getSize()

        return num_dofs

    @cached_property
    def test_functions(self):
        functions = []
        for subspace in self.function_spaces:
            functions.append(ufl.TestFunction(subspace))

        return tuple(functions)

    @cached_property
    def u(self):
        functions = []
        for name, subspace in self._mixed_space.items():
            functions.append(Function(subspace, name=name))

        return tuple(functions)

    @cached_property
    def appctx(self):
        return {"v": self.u[0]}

    @cached_property
    def F_form(self):
        v, p, T = self.u
        v_te, p_te, T_te = self.test_functions

        Ra = self.Ra
        Pr = self.Pr
        g = self.g

        F0 = (
            inner(grad(v), grad(v_te)) * dx
            + inner(dot(grad(v), v), v_te) * dx
            - inner(p, div(v_te)) * dx
            - (Ra / Pr) * inner(T * g, v_te) * dx
        )
        F1 = -div(v) * p_te * dx
        F2 = inner(dot(grad(T), v), T_te) * dx + (1.0 / Pr) * inner(grad(T), grad(T_te)) * dx

        return [F0, F1, F2]

    @cached_property
    def J_form(self):
        J_form = [[None for i in range(len(self.u))] for j in range(len(self.u))]

        for i in range(len(self.u)):
            for j in range(len(self.u)):
                J_form[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        self.F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)
                    )
                )
                if J_form[i][j].empty():
                    J_form[i][j] = None

        return J_form

    @cached_property
    def bcs(self):
        V_v, V_p, V_T = self.function_spaces
        mesh, regions, map_target_idx = self.domain_data
        facetdim = mesh.topology.dim - 1

        left_id = map_target_idx["left"]
        left_facets = np.where(regions.values == left_id)[0]
        leftdofsV_v = fem.locate_dofs_topological(V_v, facetdim, left_facets)
        leftdofsV_T = fem.locate_dofs_topological(V_T, facetdim, left_facets)

        right_id = map_target_idx["right"]
        right_facets = np.where(regions.values == right_id)[0]
        rightdofsV_v = fem.locate_dofs_topological(V_v, facetdim, right_facets)
        rightdofsV_T = fem.locate_dofs_topological(V_T, facetdim, right_facets)

        rest_id = map_target_idx["rest"]
        rest_facets = np.where(regions.values == rest_id)[0]
        restdofsV_v = fem.locate_dofs_topological(V_v, facetdim, rest_facets)

        bcs = []

        v_walls = Function(V_v)
        wallsdofsV_v = np.unique(np.hstack((leftdofsV_v, rightdofsV_v, restdofsV_v)))
        bcs.append(fem.DirichletBC(v_walls, wallsdofsV_v))

        T_left = Function(V_T)
        with T_left.vector.localForm() as T_local:
            T_local.set(1.0)
        bcs.append(fem.DirichletBC(T_left, leftdofsV_T))

        T_right = Function(V_T)
        bcs.append(fem.DirichletBC(T_right, rightdofsV_T))

        return tuple(bcs)

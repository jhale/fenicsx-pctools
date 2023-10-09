from collections import OrderedDict
from functools import cached_property

import numpy as np

import ufl
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    VectorFunctionSpace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_unit_cube,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)
from ufl import div, dot, dx, grad, inner

from petsc4py import PETSc


def get_boundary_mask(mesh):
    topology = mesh.topology
    tdim = topology.dim
    topology.create_connectivity(tdim - 1, tdim)  # mark interior/exterior facets

    imap = topology.index_map(tdim - 1)
    bndry_entities = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
    bndry_mask = np.full(imap.size_local + imap.num_ghosts, False)
    bndry_mask[bndry_entities] = True

    return bndry_mask


class Problem(object):
    def __init__(self, comm, N=10):
        self.comm = comm
        self.N = N

        self.coeffs = dict(Ra=200.0, Pr=6.8, g=(0.0, 0.0, -1.0))

    @cached_property
    def domain_data(self):
        cell_type = CellType.tetrahedron
        # cell_type = cpp.mesh.CellType.hexahedron
        ghost_mode = GhostMode.shared_facet

        mesh = create_unit_cube(
            self.comm, self.N, self.N, self.N, cell_type=cell_type, ghost_mode=ghost_mode
        )
        mesh.topology.create_connectivity(2, 2)
        imap = mesh.topology.index_map(2)
        indices = np.arange(0, imap.size_local + imap.num_ghosts)
        mesh_tags_facets = meshtags(mesh, 2, indices, np.zeros_like(indices, dtype=np.intc))
        bndry_tag_map = {"left": 1, "right": 2, "rest": 3}

        bndry_mask = get_boundary_mask(mesh)
        mesh_tags_facets.values[bndry_mask] = bndry_tag_map["rest"]

        left_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 0.0))
        mesh_tags_facets.values[left_facets] = bndry_tag_map["left"]

        right_facets = locate_entities(mesh, 2, lambda x: np.isclose(x[0], 1.0))
        mesh_tags_facets.values[right_facets] = bndry_tag_map["right"]

        return mesh, mesh_tags_facets, bndry_tag_map

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
    def F_ufl(self):
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
    def J_ufl(self):
        J_ufl = [[None for i in range(len(self.u))] for j in range(len(self.u))]

        for i in range(len(self.u)):
            for j in range(len(self.u)):
                J_ufl[i][j] = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        self.F_ufl[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)
                    )
                )
                if J_ufl[i][j].empty():
                    J_ufl[i][j] = None

        return J_ufl

    @cached_property
    def bcs(self):
        V_v, V_p, V_T = self.function_spaces
        mesh, mesh_tags_facets, bndry_tag_map = self.domain_data
        facetdim = mesh.topology.dim - 1

        left_id = bndry_tag_map["left"]
        left_facets = np.where(mesh_tags_facets.values == left_id)[0]
        leftdofsV_v = locate_dofs_topological(V_v, facetdim, left_facets)
        leftdofsV_T = locate_dofs_topological(V_T, facetdim, left_facets)

        right_id = bndry_tag_map["right"]
        right_facets = np.where(mesh_tags_facets.values == right_id)[0]
        rightdofsV_v = locate_dofs_topological(V_v, facetdim, right_facets)
        rightdofsV_T = locate_dofs_topological(V_T, facetdim, right_facets)

        rest_id = bndry_tag_map["rest"]
        rest_facets = np.where(mesh_tags_facets.values == rest_id)[0]
        restdofsV_v = locate_dofs_topological(V_v, facetdim, rest_facets)

        bcs = []

        v_walls = Function(V_v)
        wallsdofsV_v = np.unique(np.hstack((leftdofsV_v, rightdofsV_v, restdofsV_v)))
        bcs.append(dirichletbc(v_walls, wallsdofsV_v))

        T_left = Function(V_T)
        with T_left.vector.localForm() as T_local:
            T_local.set(1.0)
        bcs.append(dirichletbc(T_left, leftdofsV_T))

        T_right = Function(V_T)
        bcs.append(dirichletbc(T_right, rightdofsV_T))

        return tuple(bcs)

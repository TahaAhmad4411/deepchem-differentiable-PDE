"""Featurizers for FEM/FVM benchmark problems."""

from __future__ import annotations

from typing import Iterable, List

import torch

import deepchem as dc
from deepchem.physics.mesh import Mesh, make_unit_square_tri_mesh


class MeshFeaturizer(dc.feat.Featurizer):
    """Create structured triangular meshes from PDE specifications.

    The input datapoint is expected to be a mapping with optional keys such as
    ``nx``, ``ny``, ``dirichlet_fn``, and ``rhs_fn``.
    """

    def __init__(self, default_nx: int = 8, default_ny: int = 8):
        self.default_nx = default_nx
        self.default_ny = default_ny

    def _featurize(self, datapoint) -> object:
        nx = int(datapoint.get("nx", self.default_nx))
        ny = int(datapoint.get("ny", self.default_ny))
        mesh = make_unit_square_tri_mesh(nx, ny)
        dirichlet_fn = datapoint.get("dirichlet_fn")
        if dirichlet_fn is not None:
            boundary_nodes = mesh.dirichlet_nodes
            boundary_values = dirichlet_fn(mesh.nodes[boundary_nodes])
            mesh = Mesh(nodes=mesh.nodes,
                        elements=mesh.elements,
                        dirichlet_nodes=boundary_nodes,
                        dirichlet_values=boundary_values,
                        tags=mesh.tags)
        return {
            "mesh":
            mesh,
            "rhs_fn":
            datapoint.get("rhs_fn"),
            "conductivity":
            datapoint.get("conductivity", torch.tensor(1.0,
                                                       dtype=torch.float64)),
        }

    def featurize(self, datapoints: Iterable[object],
                  **kwargs) -> List[object]:
        return [self._featurize(x) for x in datapoints]

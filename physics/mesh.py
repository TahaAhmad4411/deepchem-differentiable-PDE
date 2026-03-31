"""Mesh data structures for differentiable FEM/FVM workflows.

This module provides a small, typed mesh representation intended for use in
PyTorch-native finite element and finite volume solvers.  The implementation is
kept deliberately minimal so it can be integrated into DeepChem without adding
heavy third-party meshing dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class Mesh:
    """A simple 2D triangular mesh.

    Parameters
    ----------
    nodes
        Node coordinates with shape ``(n_nodes, 2)``.
    elements
        Triangular connectivity with shape
        ``(n_elements, 3)``.  Each row stores integer node
        indices for one element.
    dirichlet_nodes
        Optional node indices for essential boundary conditions.
    dirichlet_values
        Optional values prescribed at ``dirichlet_nodes``.
    tags
        Optional dictionary of additional mesh metadata such as boundary masks,
        region ids, or user-defined labels.

    Notes
    -----
    The class is intentionally immutable.  Mesh adaptation or smoothing should
    produce a new ``Mesh`` instance instead of mutating an existing one.
    """

    nodes: torch.Tensor
    elements: torch.Tensor
    dirichlet_nodes: Optional[torch.Tensor] = None
    dirichlet_values: Optional[torch.Tensor] = None
    tags: Optional[Dict[str, torch.Tensor]] = None

    def validate(self) -> None:
        """Validate mesh tensor shapes and numerical consistency.

        Raises
        ------
        ValueError
            If the mesh tensors have invalid shapes, contain
            invalid indices, or represent degenerate triangles.
        """
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n_nodes, 2)")
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError("elements must have shape (n_elements, 3)")
        if not torch.is_floating_point(self.nodes):
            raise ValueError("nodes must be floating point")
        if self.elements.dtype not in (torch.int32, torch.int64):
            raise ValueError("elements must be integer typed")
        if not torch.isfinite(self.nodes).all():
            raise ValueError("nodes contains non-finite coordinates")
        if self.elements.numel() == 0:
            raise ValueError("elements must not be empty")
        if torch.min(self.elements) < 0:
            raise ValueError("elements contains negative node indices")
        if torch.max(self.elements) >= self.n_nodes:
            raise ValueError("elements contains out-of-range node indices")
        if self.dirichlet_nodes is not None:
            if self.dirichlet_values is None:
                raise ValueError(
                    "dirichlet_values must be provided"
                    " when dirichlet_nodes is set"
                )
            if self.dirichlet_nodes.ndim != 1:
                raise ValueError("dirichlet_nodes must be rank-1")
            if torch.min(self.dirichlet_nodes) < 0 or torch.max(
                    self.dirichlet_nodes) >= self.n_nodes:
                raise ValueError(
                    "dirichlet_nodes contains out-of-range node indices")
            if self.dirichlet_values.shape[0] != self.dirichlet_nodes.shape[0]:
                raise ValueError(
                    "dirichlet_values length must match dirichlet_nodes")
        areas = self.element_areas()
        if torch.any(areas <= 0):
            raise ValueError("mesh contains inverted or degenerate elements")

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        """Return the number of triangular elements."""
        return int(self.elements.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device on which the mesh is stored."""
        return self.nodes.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating point dtype used by the geometry."""
        return self.nodes.dtype

    def element_coordinates(self) -> torch.Tensor:
        """Return coordinates for each element.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(n_elements, 3, 2)``.
        """
        return self.nodes[self.elements]

    def element_areas(self) -> torch.Tensor:
        """Compute signed areas for all triangular elements.

        Returns
        -------
        torch.Tensor
            Positive area of each triangle with shape ``(n_elements,)``.
        """
        coords = self.element_coordinates()
        v0 = coords[:, 1] - coords[:, 0]
        v1 = coords[:, 2] - coords[:, 0]
        twice_area = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        return 0.5 * twice_area

    def free_nodes(self) -> torch.Tensor:
        """Return unconstrained node indices.

        Returns
        -------
        torch.Tensor
            Rank-1 tensor of unconstrained node indices.
        """
        if self.dirichlet_nodes is None:
            return torch.arange(self.n_nodes,
                                device=self.device,
                                dtype=torch.long)
        fixed = torch.zeros(self.n_nodes, device=self.device, dtype=torch.bool)
        fixed[self.dirichlet_nodes] = True
        return torch.arange(self.n_nodes, device=self.device,
                            dtype=torch.long)[~fixed]

    def with_nodes(self, nodes: torch.Tensor) -> "Mesh":
        """Create a new mesh with updated node coordinates.

        Parameters
        ----------
        nodes
            Updated node coordinates with the same shape as ``self.nodes``.

        Returns
        -------
        Mesh
            New mesh instance.
        """
        return Mesh(nodes=nodes,
                    elements=self.elements,
                    dirichlet_nodes=self.dirichlet_nodes,
                    dirichlet_values=self.dirichlet_values,
                    tags=self.tags)


def make_unit_square_tri_mesh(nx: int,
                              ny: int,
                              *,
                              dtype: torch.dtype = torch.float64,
                              device: Optional[torch.device] = None) -> Mesh:
    """Generate a uniform right-triangular mesh on the unit square.

    Parameters
    ----------
    nx
        Number of cells along the ``x`` axis.
    ny
        Number of cells along the ``y`` axis.
    dtype
        Floating point dtype for the coordinates.
    device
        Device for returned tensors.

    Returns
    -------
    Mesh
        A validated triangular mesh with Dirichlet boundary
        nodes initialized to zero values on the full boundary.
    """
    xs = torch.linspace(0.0, 1.0, nx + 1, dtype=dtype, device=device)
    ys = torch.linspace(0.0, 1.0, ny + 1, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])
    elements_t = torch.tensor(elements, dtype=torch.long, device=device)

    boundary_mask = ((nodes[:, 0] == 0.0) | (nodes[:, 0] == 1.0) |
                     (nodes[:, 1] == 0.0) | (nodes[:, 1] == 1.0))
    dirichlet_nodes = torch.nonzero(boundary_mask, as_tuple=False).reshape(-1)
    dirichlet_values = torch.zeros(dirichlet_nodes.shape[0],
                                   dtype=dtype,
                                   device=device)
    mesh = Mesh(nodes=nodes,
                elements=elements_t,
                dirichlet_nodes=dirichlet_nodes,
                dirichlet_values=dirichlet_values,
                tags={"boundary_mask": boundary_mask})
    mesh.validate()
    return mesh

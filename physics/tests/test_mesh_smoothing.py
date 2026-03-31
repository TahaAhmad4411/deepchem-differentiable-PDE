"""Tests for mesh smoothing methods."""

from __future__ import annotations

import pytest
import torch

from deepchem.physics.mesh import make_unit_square_tri_mesh
from deepchem.physics.mesh_smoothing import (
    laplace_smooth, poisson_smooth, winslow_smooth
)


@pytest.mark.torch
def test_laplace_smoothing_preserves_boundary() -> None:
    mesh = make_unit_square_tri_mesh(6, 6)
    smoothed = laplace_smooth(mesh, n_steps=1)
    assert torch.allclose(smoothed.nodes[mesh.dirichlet_nodes],
                          mesh.nodes[mesh.dirichlet_nodes])
    assert torch.equal(smoothed.elements, mesh.elements)


@pytest.mark.torch
def test_poisson_smoothing_preserves_connectivity() -> None:
    mesh = make_unit_square_tri_mesh(6, 6)
    px = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    py = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    smoothed = poisson_smooth(mesh, px=px, py=py, n_steps=1)
    assert torch.equal(smoothed.elements, mesh.elements)


@pytest.mark.torch
def test_winslow_smoothing_returns_valid_mesh() -> None:
    mesh = make_unit_square_tri_mesh(6, 6)
    smoothed = winslow_smooth(mesh, n_iters=2)
    smoothed.validate()

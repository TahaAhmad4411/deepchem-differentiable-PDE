"""Tests for mesh utilities."""

from __future__ import annotations

import pytest
import torch

from deepchem.physics.mesh import Mesh, make_unit_square_tri_mesh


@pytest.mark.torch
def test_make_unit_square_mesh() -> None:
    mesh = make_unit_square_tri_mesh(4, 4)
    assert mesh.n_nodes == 25
    assert mesh.n_elements == 32
    mesh.validate()


@pytest.mark.torch
def test_invalid_element_index_raises() -> None:
    mesh = make_unit_square_tri_mesh(2, 2)
    bad = Mesh(nodes=mesh.nodes,
               elements=torch.tensor([[0, 1, 99]], dtype=torch.long),
               dirichlet_nodes=mesh.dirichlet_nodes,
               dirichlet_values=mesh.dirichlet_values)
    with pytest.raises(ValueError):
        bad.validate()

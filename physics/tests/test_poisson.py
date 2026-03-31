"""Tests for differentiable Poisson FEM."""

from __future__ import annotations

import math

import pytest
import torch
from torch.autograd import gradcheck

from deepchem.physics.fem.poisson import l2_error, solve_poisson
from deepchem.physics.mesh import Mesh, make_unit_square_tri_mesh


def exact_solution(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def forcing_term(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def build_mesh(nx: int) -> Mesh:
    mesh = make_unit_square_tri_mesh(nx, nx)
    boundary_values = exact_solution(mesh.nodes[mesh.dirichlet_nodes])
    return Mesh(nodes=mesh.nodes,
                elements=mesh.elements,
                dirichlet_nodes=mesh.dirichlet_nodes,
                dirichlet_values=boundary_values,
                tags=mesh.tags)


@pytest.mark.torch
def test_poisson_solution_is_accurate() -> None:
    mesh = build_mesh(16)
    u = solve_poisson(mesh, torch.tensor(1.0, dtype=torch.float64),
                      forcing_term)
    error = l2_error(mesh, u, exact_solution)
    assert float(error) < 2.0e-2


@pytest.mark.torch
def test_conductivity_gradientcheck() -> None:
    mesh = build_mesh(4)

    def objective(k: torch.Tensor) -> torch.Tensor:
        sol = solve_poisson(mesh, k.squeeze(), forcing_term)
        return torch.sum(sol**2)

    k0 = torch.tensor([1.1], dtype=torch.float64, requires_grad=True)
    assert gradcheck(objective, (k0, ), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.torch
def test_inverse_problem_reduces_loss() -> None:
    mesh = build_mesh(8)
    true_k = torch.tensor(1.5, dtype=torch.float64)
    with torch.no_grad():
        target = solve_poisson(mesh, true_k, forcing_term)
    log_k = torch.tensor(math.log(0.7),
                         dtype=torch.float64,
                         requires_grad=True)
    optimizer = torch.optim.Adam([log_k], lr=5e-2)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        pred = solve_poisson(mesh, torch.exp(log_k), forcing_term)
        loss = torch.mean((pred - target)**2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]

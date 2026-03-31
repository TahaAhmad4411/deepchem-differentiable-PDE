"""Benchmarks for differentiable FEM workflows.

Run this file from a DeepChem source checkout.  The script demonstrates both
h-convergence and gradient-based inverse recovery.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from deepchem.physics.fem.poisson import l2_error, solve_poisson
from deepchem.physics.mesh import Mesh, make_unit_square_tri_mesh


def exact_solution(coords: torch.Tensor) -> torch.Tensor:
    """Manufactured solution on the unit square."""
    x = coords[:, 0]
    y = coords[:, 1]
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def forcing_term(coords: torch.Tensor) -> torch.Tensor:
    """Right-hand side corresponding to :func:`exact_solution`."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def build_mesh(nx: int) -> Mesh:
    """Build a benchmark mesh with exact Dirichlet data."""
    mesh = make_unit_square_tri_mesh(nx, nx)
    boundary_values = exact_solution(mesh.nodes[mesh.dirichlet_nodes])
    return Mesh(nodes=mesh.nodes,
                elements=mesh.elements,
                dirichlet_nodes=mesh.dirichlet_nodes,
                dirichlet_values=boundary_values,
                tags=mesh.tags)


def run_h_convergence() -> List[Tuple[int, float, float, float]]:
    """Run an h-convergence study for P1 triangles."""
    sizes = [4, 8, 16, 32]
    rows = []
    prev_error = None
    for nx in sizes:
        mesh = build_mesh(nx)
        u = solve_poisson(mesh, torch.tensor(1.0, dtype=torch.float64),
                          forcing_term)
        error = float(l2_error(mesh, u, exact_solution).item())
        h = 1.0 / nx
        rate = float("nan")
        if prev_error is not None:
            rate = math.log(prev_error / error) / math.log(2.0)
        rows.append((nx, h, error, rate))
        prev_error = error
    return rows


def run_inverse_problem(steps: int = 50) -> Tuple[float, float, float, float]:
    """Run a scalar conductivity inverse problem benchmark."""
    mesh = build_mesh(8)
    true_k = torch.tensor(1.7, dtype=torch.float64)
    with torch.no_grad():
        target = solve_poisson(mesh, true_k, forcing_term)
    log_k = torch.tensor(math.log(0.8),
                         dtype=torch.float64,
                         requires_grad=True)
    optimizer = torch.optim.Adam([log_k], lr=5e-2)

    initial_loss = None
    initial_grad = None
    final_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        pred = solve_poisson(mesh, torch.exp(log_k), forcing_term)
        loss = torch.mean((pred - target)**2)
        loss.backward()
        if step == 0:
            initial_loss = float(loss.item())
            initial_grad = float(log_k.grad.abs().item())
        optimizer.step()
        final_loss = float(loss.item())
    return initial_grad, initial_loss, final_loss, float(
        torch.exp(log_k).item())


def main() -> None:
    """Execute the benchmark suite and print results."""
    print("=" * 55)
    print("FEMSolver h-convergence benchmark")
    print("Problem: -∇²u = 2π²sin(πx)sin(πy), u_exact = sin(πx)sin(πy)")
    print("=" * 55)
    print(f"{'nx':>6} {'h':>10} {'L2_error':>14} {'rate':>10}")
    print("-" * 55)
    rows = run_h_convergence()
    for nx, h, error, rate in rows:
        rate_text = "—" if math.isnan(rate) else f"{rate:.2f}"
        print(f"{nx:6d} {h:10.4f} {error:14.6f} {rate_text:>10}")
    print("=" * 55)
    print(f"Convergence rate (last refinement): {rows[-1][3]:.2f}")
    print("Expected for P1 elements:           ~2.00")
    print()

    print("=" * 55)
    print("Differentiability benchmark (inverse Poisson)")
    print("Mesh: 8x8")
    print("=" * 55)
    grad_norm, initial_loss, final_loss, recovered_k = run_inverse_problem()
    print(f"Gradient norm at step 0: {grad_norm:.6f}")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss ({50} steps): {final_loss:.6f}")
    print(f"Recovered conductivity: {recovered_k:.6f}")
    print(f"Loss reduction: {100.0 * (1.0 - final_loss / initial_loss):.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()

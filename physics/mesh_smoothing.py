"""Topology-preserving differentiable mesh smoothing.

The routines in this module move interior mesh nodes while preserving the fixed
triangular connectivity.  This is the class of mesh motion that is most
compatible with autograd-based PDE workflows.
"""

from __future__ import annotations

from typing import Optional

import torch

from deepchem.physics.mesh import Mesh


def _graph_laplacian(mesh: Mesh) -> torch.Tensor:
    """Build a simple graph Laplacian for mesh nodes.

    Parameters
    ----------
    mesh
        Input triangular mesh.

    Returns
    -------
    torch.Tensor
        Dense graph Laplacian of shape ``(n_nodes, n_nodes)``.
    """
    A = torch.zeros((mesh.n_nodes, mesh.n_nodes),
                    dtype=mesh.dtype,
                    device=mesh.device)
    for elem in mesh.elements:
        for i in range(3):
            for j in range(i + 1, 3):
                ni = int(elem[i])
                nj = int(elem[j])
                A[ni, nj] += 1.0
                A[nj, ni] += 1.0
    degree = torch.sum(A, dim=1)
    return torch.diag(degree) - A


def laplace_smooth(mesh: Mesh, n_steps: int = 1) -> Mesh:
    """Apply Laplacian smoothing to interior nodes.

    Parameters
    ----------
    mesh
        Input mesh.
    n_steps
        Number of smoothing iterations.

    Returns
    -------
    Mesh
        Smoothed mesh with identical connectivity.
    """
    new_mesh = mesh
    for _ in range(n_steps):
        L = _graph_laplacian(new_mesh)
        free = new_mesh.free_nodes()
        if free.numel() == 0:
            return new_mesh
        fixed = new_mesh.dirichlet_nodes
        x = new_mesh.nodes[:, 0]
        y = new_mesh.nodes[:, 1]
        L_ff = L[free][:, free]
        rhs_x = -L[free][:, fixed] @ x[fixed]
        rhs_y = -L[free][:, fixed] @ y[fixed]
        x_new = x.clone()
        y_new = y.clone()
        x_new[free] = torch.linalg.solve(L_ff, rhs_x)
        y_new[free] = torch.linalg.solve(L_ff, rhs_y)
        new_mesh = new_mesh.with_nodes(torch.stack([x_new, y_new], dim=1))
    new_mesh.validate()
    return new_mesh


def poisson_smooth(mesh: Mesh,
                   px: Optional[torch.Tensor] = None,
                   py: Optional[torch.Tensor] = None,
                   n_steps: int = 1) -> Mesh:
    """Apply Poisson smoothing with optional source terms.

    Parameters
    ----------
    mesh
        Input mesh.
    px
        Optional source term for the x-coordinate equation.
    py
        Optional source term for the y-coordinate equation.
    n_steps
        Number of smoothing iterations.

    Returns
    -------
    Mesh
        Smoothed mesh.
    """
    new_mesh = mesh
    for _ in range(n_steps):
        L = _graph_laplacian(new_mesh)
        free = new_mesh.free_nodes()
        if free.numel() == 0:
            return new_mesh
        fixed = new_mesh.dirichlet_nodes
        x = new_mesh.nodes[:, 0]
        y = new_mesh.nodes[:, 1]
        L_ff = L[free][:, free]
        src_x = torch.zeros_like(free, dtype=mesh.dtype).to(mesh.device)
        src_y = torch.zeros_like(free, dtype=mesh.dtype).to(mesh.device)
        if px is not None:
            src_x = px[free]
        if py is not None:
            src_y = py[free]
        rhs_x = src_x - L[free][:, fixed] @ x[fixed]
        rhs_y = src_y - L[free][:, fixed] @ y[fixed]
        x_new = x.clone()
        y_new = y.clone()
        x_new[free] = torch.linalg.solve(L_ff, rhs_x)
        y_new[free] = torch.linalg.solve(L_ff, rhs_y)
        new_mesh = new_mesh.with_nodes(torch.stack([x_new, y_new], dim=1))
    new_mesh.validate()
    return new_mesh


def winslow_smooth(mesh: Mesh, n_iters: int = 5, eps: float = 1e-8) -> Mesh:
    """Apply a simple fixed-point Winslow-style smoothing iteration.

    Parameters
    ----------
    mesh
        Input mesh.
    n_iters
        Number of fixed-point iterations.
    eps
        Small positive value used to stabilize reciprocal edge lengths.

    Returns
    -------
    Mesh
        Smoothed mesh with preserved topology.
    """
    current = mesh
    for _ in range(n_iters):
        A = torch.zeros((current.n_nodes, current.n_nodes),
                        dtype=current.dtype,
                        device=current.device)
        for elem in current.elements:
            idx = elem.tolist()
            for i, j in ((idx[0], idx[1]), (idx[1], idx[2]), (idx[2], idx[0])):
                dist = torch.linalg.norm(current.nodes[i] - current.nodes[j])
                w = 1.0 / (dist + eps)
                A[i, j] += w
                A[j, i] += w
        degree = torch.sum(A, dim=1)
        L = torch.diag(degree) - A
        free = current.free_nodes()
        if free.numel() == 0:
            return current
        fixed = current.dirichlet_nodes
        x = current.nodes[:, 0]
        y = current.nodes[:, 1]
        L_ff = L[free][:, free]
        rhs_x = -L[free][:, fixed] @ x[fixed]
        rhs_y = -L[free][:, fixed] @ y[fixed]
        x_new = x.clone()
        y_new = y.clone()
        x_new[free] = torch.linalg.solve(L_ff, rhs_x)
        y_new[free] = torch.linalg.solve(L_ff, rhs_y)
        current = current.with_nodes(torch.stack([x_new, y_new], dim=1))
    current.validate()
    return current

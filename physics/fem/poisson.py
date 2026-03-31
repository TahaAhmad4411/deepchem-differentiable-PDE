"""Differentiable Poisson FEM utilities.

The functions in this module implement a small P1 finite element solver for the
scalar Poisson equation on 2D triangular meshes.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from deepchem.physics.mesh import Mesh

TensorLikeOrCallable = Union[torch.Tensor, Callable[[torch.Tensor],
                                                    torch.Tensor]]


def local_stiffness_matrix(coords: torch.Tensor,
                           conductivity: torch.Tensor) -> torch.Tensor:
    """Compute the local stiffness matrix for P1 triangular elements.

    Parameters
    ----------
    coords
        Element coordinates with shape ``(n_elements, 3, 2)``.
    conductivity
        Elementwise conductivity with shape ``(n_elements,)`` or scalar tensor.

    Returns
    -------
    torch.Tensor
        Local stiffness matrices with shape ``(n_elements, 3, 3)``.
    """
    x1 = coords[:, 0, 0]
    y1 = coords[:, 0, 1]
    x2 = coords[:, 1, 0]
    y2 = coords[:, 1, 1]
    x3 = coords[:, 2, 0]
    y3 = coords[:, 2, 1]

    twice_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * twice_area
    b = torch.stack([y2 - y3, y3 - y1, y1 - y2], dim=1)
    c = torch.stack([x3 - x2, x1 - x3, x2 - x1], dim=1)

    grad_grad = b.unsqueeze(2) * b.unsqueeze(1) + c.unsqueeze(2) * c.unsqueeze(
        1)
    coeff = conductivity.reshape(-1, 1, 1) / (4.0 * area).reshape(-1, 1, 1)
    return coeff * grad_grad


def assemble_stiffness(mesh: Mesh, conductivity: torch.Tensor) -> torch.Tensor:
    """Assemble the dense global stiffness matrix.

    Parameters
    ----------
    mesh
        Valid triangular mesh.
    conductivity
        Scalar tensor or elementwise conductivity of shape ``(n_elements,)``.

    Returns
    -------
    torch.Tensor
        Dense global stiffness matrix with shape ``(n_nodes, n_nodes)``.
    """
    if conductivity.ndim == 0:
        conductivity = conductivity.expand(mesh.n_elements)
    coords = mesh.element_coordinates()
    ke = local_stiffness_matrix(coords, conductivity)
    K = torch.zeros((mesh.n_nodes, mesh.n_nodes),
                    dtype=mesh.dtype,
                    device=mesh.device)
    for elem, elem_k in zip(mesh.elements, ke):
        K[elem[:, None], elem[None, :]] += elem_k
    return K


def assemble_load(mesh: Mesh, rhs: TensorLikeOrCallable) -> torch.Tensor:
    """Assemble the load vector using a centroid quadrature rule.

    Parameters
    ----------
    mesh
        Valid triangular mesh.
    rhs
        Right-hand side evaluated either as nodal values with shape
        ``(n_nodes,)`` or a callable accepting element centroids ``(n_elements,
        2)`` and returning elementwise values.

    Returns
    -------
    torch.Tensor
        Global load vector with shape ``(n_nodes,)``.
    """
    if callable(rhs):
        coords = mesh.element_coordinates()
        centroids = torch.mean(coords, dim=1)
        values = rhs(centroids)
    else:
        if rhs.shape != (mesh.n_nodes, ):
            raise ValueError("rhs tensor must have shape (n_nodes,)")
        values = torch.mean(rhs[mesh.elements], dim=1)

    areas = mesh.element_areas()
    element_load = (areas * values / 3.0).unsqueeze(1).expand(-1, 3)
    b = torch.zeros(mesh.n_nodes, dtype=mesh.dtype, device=mesh.device)
    for elem, elem_b in zip(mesh.elements, element_load):
        b[elem] += elem_b
    return b


def apply_dirichlet(
    K: torch.Tensor, b: torch.Tensor, dirichlet_nodes: Optional[torch.Tensor],
    dirichlet_values: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Dirichlet boundary conditions to a linear system.

    Parameters
    ----------
    K
        Stiffness matrix with shape ``(n_nodes, n_nodes)``.
    b
        Load vector with shape ``(n_nodes,)``.
    dirichlet_nodes
        Rank-1 tensor of constrained node indices.
    dirichlet_values
        Rank-1 tensor of prescribed values.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Modified matrix and vector.
    """
    if dirichlet_nodes is None or dirichlet_values is None:
        return K, b
    K_mod = K.clone()
    b_mod = b.clone()
    for node, value in zip(dirichlet_nodes, dirichlet_values):
        b_mod = b_mod - K_mod[:, node] * value
        K_mod[node, :] = 0.0
        K_mod[:, node] = 0.0
        K_mod[node, node] = 1.0
        b_mod[node] = value
    return K_mod, b_mod


def solve_poisson(mesh: Mesh, conductivity: torch.Tensor,
                  rhs: TensorLikeOrCallable) -> torch.Tensor:
    """Solve ``-div(k grad u) = f`` on a triangular mesh.

    Parameters
    ----------
    mesh
        Valid triangular mesh.
    conductivity
        Scalar tensor or elementwise conductivity values.
    rhs
        Nodal values or callable right-hand side.

    Returns
    -------
    torch.Tensor
        FEM solution vector with shape ``(n_nodes,)``.
    """
    mesh.validate()
    K = assemble_stiffness(mesh, conductivity)
    b = assemble_load(mesh, rhs)
    K_bc, b_bc = apply_dirichlet(K, b, mesh.dirichlet_nodes,
                                 mesh.dirichlet_values)
    return torch.linalg.solve(K_bc, b_bc)


def l2_error(mesh: Mesh, u_pred: torch.Tensor,
             u_exact: TensorLikeOrCallable) -> torch.Tensor:
    """Compute an elementwise ``L2`` error estimate.

    Parameters
    ----------
    mesh
        Valid triangular mesh.
    u_pred
        Predicted nodal values.
    u_exact
        Exact nodal values or callable taking node coordinates.

    Returns
    -------
    torch.Tensor
        Scalar ``L2`` error.
    """
    if callable(u_exact):
        exact = u_exact(mesh.nodes)
    else:
        exact = u_exact
    diff = u_pred - exact
    areas = mesh.element_areas()
    diff_sq = torch.mean(diff[mesh.elements]**2, dim=1)
    return torch.sqrt(torch.sum(areas * diff_sq))

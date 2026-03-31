"""DeepChem TorchModel wrapper for differentiable FEM inverse problems."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

import deepchem as dc
from deepchem.models.losses import L2Loss
from deepchem.physics.fem.poisson import solve_poisson


class _InversePoissonModule(torch.nn.Module):
    """Small trainable inverse-problem module.

    The module learns a scalar conductivity value from observations of the
    solution field.
    """

    def __init__(self, initial_conductivity: float = 1.0):
        super().__init__()
        self.log_conductivity = torch.nn.Parameter(
            torch.tensor(np.log(initial_conductivity), dtype=torch.float64))

    def forward(self, inputs: List[Dict[str, object]]) -> torch.Tensor:
        outputs = []
        conductivity = torch.exp(self.log_conductivity)
        for item in inputs:
            mesh = item["mesh"]
            rhs_fn = item["rhs_fn"]
            outputs.append(solve_poisson(mesh, conductivity, rhs_fn))
        return torch.stack(outputs, dim=0)


class FEMModel(dc.models.TorchModel):
    """A ``TorchModel`` wrapper for differentiable Poisson inverse problems.

    Parameters
    ----------
    initial_conductivity
        Initial scalar conductivity parameter.
    learning_rate
        Optimizer learning rate.
    model_dir
        Optional model directory.
    """

    def __init__(self,
                 initial_conductivity: float = 1.0,
                 learning_rate: float = 1e-2,
                 model_dir: str | None = None,
                 **kwargs):
        module = _InversePoissonModule(
            initial_conductivity=initial_conductivity)
        super().__init__(module,
                         loss=L2Loss(),
                         learning_rate=learning_rate,
                         model_dir=model_dir,
                         **kwargs)

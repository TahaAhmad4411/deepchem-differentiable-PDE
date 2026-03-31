"""Dataset helpers for differentiable PDE problems."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

from deepchem.data import NumpyDataset


class MeshDataset(NumpyDataset):
    """A thin ``NumpyDataset`` wrapper for mesh-based PDE examples.

    Parameters
    ----------
    X
        Sequence of mesh problem specifications.  Individual entries may be
        Python objects such as dictionaries containing ``mesh``, ``rhs``, and
        ``conductivity`` tensors.
    y
        Optional target outputs.
    w
        Optional sample weights.
    ids
        Optional ids.
    """

    def __init__(self,
                 X: Sequence[object],
                 y: Optional[Sequence[object]] = None,
                 w: Optional[Sequence[object]] = None,
                 ids: Optional[Sequence[object]] = None):
        x_arr = np.asarray(list(X), dtype=object)
        y_arr = None if y is None else np.asarray(list(y), dtype=object)
        w_arr = None if w is None else np.asarray(list(w))
        ids_arr = None if ids is None else np.asarray(list(ids), dtype=object)
        super().__init__(X=x_arr, y=y_arr, w=w_arr, ids=ids_arr)

    def iterbatches(
        self,
        batch_size: Optional[int] = None,
        epochs: int = 1,
        deterministic: bool = False,
        pad_batches: bool = False
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over minibatches.

        This method is inherited from ``NumpyDataset`` but kept here explicitly
        to make the contract visible to contributors working on TorchModel
        integration.
        """
        return super().iterbatches(batch_size=batch_size,
                                   epochs=epochs,
                                   deterministic=deterministic,
                                   pad_batches=pad_batches)

import torch
import torch.nn as nn
from torch.types import Size
from abc import ABC
import geoopt # type: ignore
import logging
from typing import Tuple, Optional
from .utils import ManifoldType

logger = logging.getLogger(__name__)

class UnitaryTensor(nn.Module):
    def __init__(
        self,
        u: torch.Tensor,
        t_shape: Size | Tuple[int, ...] | None = None,
        manifold: ManifoldType = ManifoldType.CANONICAL,
    ):
        super().__init__()
        if manifold == ManifoldType.EXACT:
            mf = geoopt.EuclideanStiefelExact()
        elif manifold == ManifoldType.FROBENIUS:
            mf = geoopt.EuclideanStiefel()
        elif manifold == ManifoldType.CANONICAL:
            mf = geoopt.CanonicalStiefel()
        elif manifold == ManifoldType.SPHERICAL:
            mf = geoopt.Sphere()
        else:
            raise ValueError(f"Invalid manifold type: {manifold}")

        self.manifold = mf
        self.weight = u.detach().clone()
        if t_shape is not None:
            self.t_shape = Size(t_shape)
        else:
            self.t_shape = Size(u.shape)

    @property
    def tensor(self) -> torch.Tensor:
        return self.weight.reshape(self.t_shape)
    
    def _to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        if device is None and dtype is None:
            raise ValueError("At least one of 'device' or 'dtype' must be provided to .to().")
        self.weight = self.weight.to(device=device, dtype=dtype).detach()

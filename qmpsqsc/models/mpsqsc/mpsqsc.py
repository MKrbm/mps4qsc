from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import math
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from .mpsbase import MPSBase
from .mpstate import MPState

class MpsQsc(MPSBase):
    """
    MPS for classification with 2 output channels by default.
    """
    def __init__(
        self,
        L: int,
        chi: int,
        d: int,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.complex128,
        init: str = "random",
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
        out_dim: int = 2,            # allow general C if desired
    ):
        super().__init__(
            L=L, chi=chi, d=d, out_dim=out_dim,
            As=As, device=device, dtype=dtype,
            init=init, seed=seed, optimize=optimize,
            requires_grad=True,
        )

    # Convenience—matches your old scoring contraction (no conjugate on weights)
    def contract_with_state(self, state: MPState) -> torch.Tensor:
        """
        Return scores y of shape (out_dim,) via raw contraction
        (i.e., no conjugation of the classifier weights).
        """
        return self.overlap(state, conjugate_self=False)


    def _clone_with_As(self, As_new: Sequence[torch.Tensor], new_chi: int | None = None) -> "MpsQsc":
        return MpsQsc(
            L=self.L,
            chi=new_chi if new_chi is not None else self.chi,
            d=self.d,
            As=[A.clone().detach() for A in As_new],
            device=self.device,
            dtype=self.dtype,
            optimize=self.optimize,
            out_dim=self.out_dim
        )
    
    def _initialize(self, init: str = "stacked", seed: Optional[int] = None) -> List[torch.Tensor]:
        if init not in {"stacked"}:
            raise ValueError("init must be 'stacked'")
        MPS_list = []
        L = self.L
        chi = self.chi
        d = self.d
        dtype = self.dtype
        out_dim = self.out_dim
        std = 1e-3
        for i in range(L):
            if i == 0:
                core = torch.zeros(d, chi, dtype=dtype)
                core[:] = 1
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            elif i == L - 1:
                core = torch.zeros(chi, d, out_dim, dtype=dtype)
                min_dim = min(chi, d)
                core[:min_dim, :min_dim] = torch.eye(min_dim, dtype=dtype)
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            else:
                core = torch.stack([torch.eye(chi, dtype=dtype)] * d).permute(1, 0, 2)
                core += torch.normal(mean=0.0, std=std, size=core.shape)
            MPS_list.append(core)
        return MPS_list

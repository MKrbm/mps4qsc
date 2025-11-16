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

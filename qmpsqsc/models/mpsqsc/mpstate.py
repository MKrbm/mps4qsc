from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import math
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from .mpsbase import MPSBase

class MPState(MPSBase):
    """
    Standard MPS state (no classifier leg). Keeps old API names where practical.
    """
    def __init__(
        self,
        L: int,
        d: int,
        chi: int | Sequence[int] | None = None,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
        init: str = "random",
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
    ):
        super().__init__(
            L=L, d=d, out_dim=1, chi=chi,
            As=As, device=device, dtype=dtype,
            init=init, seed=seed, optimize=optimize,
        )

    # Back-compat: ⟨other|self⟩ API from your previous code
    def inner_product(self, other: "MPState") -> torch.Tensor:
        """
        Backward-compatible alias: returns ⟨other|self⟩ (scalar).
        Prefer: `other.overlap(self)` (which is the same).
        """
        norm1 = self.norm()
        norm2 = other.norm()
        return other.overlap(self, conjugate_self=True) / (norm1 * norm2)

    def _clone_with_As(self, As_new: Sequence[torch.Tensor], new_chi: int | None = None) -> "MPState":
        return MPState(
            L=self.L,
            d=self.d,
            As=[A.clone().detach() for A in As_new],
            device=self.device,
            dtype=self.dtype,
            optimize=self.optimize
        )
    
    def set_requires_grad(self, requires_grad: bool = True):
        for A in self.As:
            A.requires_grad_(requires_grad)
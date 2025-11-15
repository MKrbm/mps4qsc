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
        chi: int,
        d: int,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        init: str = "random",
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
    ):
        super().__init__(
            L=L, chi=chi, d=d, out_dim=1,
            As=As, device=device, dtype=dtype,
            init=init, seed=seed, optimize=optimize,
            requires_grad=False,
        )

    # Back-compat: ⟨other|self⟩ API from your previous code
    def inner_product(self, other: "MPState") -> torch.Tensor:
        """
        Backward-compatible alias: returns ⟨other|self⟩ (scalar).
        Prefer: `other.overlap(self)` (which is the same).
        """
        return other.overlap(self, conjugate_self=True)

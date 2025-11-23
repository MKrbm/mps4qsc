from __future__ import annotations
import os
from typing import List, Optional, Sequence, Tuple
import math
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from .mpsbase import MPSBase
from .mpstate import MPState
from pathlib import Path
class MpsQsc(MPSBase):
    """
    MPS for classification with 2 output channels by default.
    """
    def __init__(
        self,
        L: int,
        d: int,
        chi: int | Sequence[int] | None = None,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.complex128,
        init: str = "random",
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
        out_dim: int = 2,            # allow general C if desired
    ):
        super().__init__(
            L=L, d=d, out_dim=out_dim, chi=chi,
            As=As, device=device, dtype=dtype,
            init=init, seed=seed, optimize=optimize,
        )

        self.D = torch.tensor(0.5, device=self.device, dtype=self.dtype)
        self.D.requires_grad_(True)

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
        chi = self.chi_max
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
    
    @classmethod
    def load_model(cls, path: os.PathLike) -> MpsQsc:
        """
        Load a saved MpsQsc model from disk.

        Parameters
        ----------
        path : os.PathLike
            Path to the file containing the saved model.

        Returns
        -------
        MpsQsc
            The loaded MpsQsc instance with state restored from file.

        Raises
        ------
        ValueError
            If the file does not contain a compatible MpsQsc model.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu")

        fmt = payload.get("format", None)
        if fmt != "mps_v2":
            raise ValueError(f"Unsupported MPS format '{fmt}' in file '{path}'.")

        class_name: str = payload.get("class_name", "MPSBase")
        if class_name != "MpsQsc":
            raise ValueError(f"Expected MpsQsc, got {class_name} in file '{path}'.")

        L: int = payload["L"]
        d: int = payload["d"]
        out_dim: int = payload.get("out_dim", 1)
        optimize: str = payload.get("optimize", "random-greedy")
        As_saved: list[torch.Tensor] = payload["As"]

        # Decide target device / dtype
        # Use saved dtype of first core
        dtype = As_saved[0].dtype
        As = [A.to(dtype) for A in As_saved]

        # Recreate the object. We assume subclasses keep the same ctor signature.
        mps: MpsQsc = MpsQsc(
            L=L,
            d=d,
            out_dim=out_dim,
            As=As,
            dtype=dtype,
            optimize=optimize,
        )
        return mps
    
    def predict(self, states: List[MPState], eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        amps_list = [self.contract_with_state(st) for st in states]  # each (2,)
        amps_batch = torch.stack(amps_list, dim=0)                  # (B, 2)
        abs_sq = (amps_batch.conj() * amps_batch).real
        denom = abs_sq.sum(dim=-1, keepdim=True).clamp_min(eps)
        return abs_sq / denom, denom.squeeze(-1)
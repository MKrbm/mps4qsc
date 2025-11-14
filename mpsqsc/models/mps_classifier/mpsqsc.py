from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import math
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from ..opt_einsum_utils import GetSymbolFn, _EqAndPathCache
from ..mpstate import MPState


def _validate_shapes_mps_qsc(L: int, d: int, chi: int, As: Sequence[torch.Tensor]):
    if len(As) != L:
        raise ValueError(f"As must have length L={L}, got {len(As)}.")
    if L < 2:
        raise ValueError("MpsQsc requires L >= 2 (first, interior..., last(class)).")
    # First
    if tuple(As[0].shape) != (d, chi):
        raise ValueError(f"As[0] must have shape (d, chi)=({d},{chi}), "
                         f"got {tuple(As[0].shape)}.")
    # Interiors
    for i in range(1, L - 1):
        if tuple(As[i].shape) != (chi, d, chi):
            raise ValueError(f"As[{i}] must have shape (chi, d, chi)=({chi},{d},{chi}), "
                             f"got {tuple(As[i].shape)}.")
    # Last
    if tuple(As[L - 1].shape) != (chi, d, 2):
        raise ValueError(f"As[L-1] must have shape (chi, d, 2)=({chi},{d},2), "
                         f"got {tuple(As[L-1].shape)}.")


class MpsQsc:
    """
    MPS for binary classification.

    Shapes (L sites):
      As[0]     : (d, chi)
      As[1..L-2]: (chi, d, chi)
      As[L-1]   : (chi, d, 2)

    contract() returns a tensor of shape (d, d, ..., d, 2).

    The contraction is implemented with opt_einsum and the optimal (or
    greedy) path is cached after the first call and reused thereafter.
    """
    def __init__(
        self,
        L: int,
        chi: int,
        d: int,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        init: str = "random",             # or "stacked_identity"
        seed: Optional[int] = None,
        optimize: str = "random-greedy"         # or "greedy"
    ):
        if L < 2:
            raise ValueError("MpsQsc requires L >= 2.")
        self.L = L
        self.chi = chi
        self.d = d
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.optimize = optimize

        self.As: List[torch.Tensor] = []
        self.symbols: List[List[str]] = []
        self._cache_fs = _EqAndPathCache() # full state cache
        self._cache_cls = _EqAndPathCache()    # For contraction with MPState for classification.
        self._sym = GetSymbolFn()

        if As is not None:
            self.set_As([a.to(self.device, self.dtype) for a in As])
        else:
            self.initialize(init=init, seed=seed)

    # ---------- Initialization ----------
    def initialize(self, init: str = "random", seed: Optional[int] = None):
        """
        Initialize tensors with either:
          - "random": normal(0, std) with scale inversely proportional to fan-in
          - "stacked_identity": block-diagonal identities across the physical index
                                (see notes below)
        """
        if seed is not None:
            torch.manual_seed(seed)

        if init not in {"random"}:
            raise ValueError("init must be 'random' or 'stacked_identity'")

        d, chi, L = self.d, self.chi, self.L
        As: List[torch.Tensor] = []

        if init == "random":
            # First
            A0 = torch.randn(d, chi, device=self.device, dtype=self.dtype) / math.sqrt(chi)
            As.append(A0)
            # Interiors
            for _ in range(1, L - 1):
                A = torch.randn(chi, d, chi, device=self.device, dtype=self.dtype) / math.sqrt(chi * d)
                As.append(A)
            # Last (classification head)
            AL = torch.randn(chi, d, 2, device=self.device, dtype=self.dtype) / math.sqrt(chi * d)
            As.append(AL)

        self.set_As(As)
        self._cache_fs.invalidate()

    def to(self, device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None):
        """Move parameters to device/dtype."""
        if device is None and dtype is None:
            return self
        new_device = torch.device(device) if device is not None else self.device
        new_dtype = dtype if dtype is not None else self.dtype
        self.set_As([A.to(new_device, new_dtype) for A in self.As])
        self.device, self.dtype = new_device, new_dtype
        self._cache_fs.invalidate()
        return self

    def set_As(self, As: Sequence[torch.Tensor]):
        """Replace tensors and reset cached path."""
        _validate_shapes_mps_qsc(self.L, self.d, self.chi, As)
        self.As = [A.to(self.device, self.dtype).detach().clone().requires_grad_(True) for A in As]
        self._create_symbols()
        self._reset_cache()
    
    def _create_symbols(self):
        L = self.L
        sym = self._sym
        sym.reset()
        self.p_symbols = [sym() for _ in range(L)]
        self.b_symbols = [sym() for _ in range(L - 1)]
        self.c_symbol = sym()
        self.symbols = [[self.p_symbols[0], self.b_symbols[0]]]
        for i in range(1, L - 1):
            self.symbols.append([self.b_symbols[i - 1], self.p_symbols[i], self.b_symbols[i]])
        self.symbols.append([self.b_symbols[L-2], self.p_symbols[L-1], self.c_symbol])
    
    def _reset_cache(self):
        self._cache_fs.invalidate()
        self._cache_cls.invalidate()

    # ---------- Contraction ----------
    def _build_equation(self) -> str:
        """
        Build an einsum equation for:
           A0(p0,b1), A1(b1,p1,b2), ..., AL-2(bL-2,pL-2,bL-1), AL-1(bL-1,pL-1,c)
        -> p0 p1 ... pL-1 c
        """

        pieces = ["".join(self.symbols[i]) for i in range(self.L)]
        # pieces.append(self.symbols[0][0] + self.symbols[0][1])
        # for i in range(1, self.L - 1):
        #     pieces.append(self.symbols[i][0] + self.symbols[i][1] + self.symbols[i][2])
        # pieces.append(self.symbols[self.L - 1][0] + self.symbols[self.L - 1][1] + self.symbols[self.L - 1][2])
        eq = ",".join(pieces) + "->" + "".join(self.p_symbols) + self.c_symbol
        return eq
    
    def _build_path(self) -> Tuple[str, oe.Path]:
        """
        Build a path for the contraction.
        """
        shapes = tuple(tuple(a.shape) for a in self.As)
        if self._cache_fs.shape_sig != shapes:
            eq = self._build_equation()
            path, _ = oe.contract_path(eq, *self.As, optimize=self.optimize)
            self._cache_fs.eq = eq
            self._cache_fs.path = path
            self._cache_fs.shape_sig = shapes
        if self._cache_fs.eq is None or self._cache_fs.path is None:
            raise ValueError("Equation or path is not built.")
        return self._cache_fs.eq, self._cache_fs.path

    def state_vector(self) -> torch.Tensor:
        """
        Contract the whole MPS to yield a tensor with shape (d,)*L + (2,).

        NOTE: This is exponentially large in L. For real tasks you may prefer
        to evaluate strings/batches instead of materializing the full tensor.
        """
        self._build_path()
        return oe.contract(self._cache_fs.eq, *self.As, optimize=self._cache_fs.path)
    
    def _shift_symbols(self, shift: int):
        self._sym.shift_chars(shift, self.symbols, self.p_symbols, self.b_symbols, self.c_symbol)
    
    def _build_state_path(self, state: "MPState") -> Tuple[str, oe.Path]:
        """
        Build (and cache) the opt_einsum equation & path for contracting
        this MpsQsc with a given MPState.

        Topology:

          Q0(p0,b1), Q1(b1,p1,b2), ..., Q_{L-1}(b_{L-1},p_{L-1},c)
          S0(p0,a1), S1(a1,p1,a2), ..., S_{L-1}(a_{L-1},p_{L-1})

        -> c

        where we sum over all physical indices p_i and all internal
        bond indices a_i (state) and b_i (classifier).
        """
        # Shapes signature: classifier shapes + state shapes
        shapes_q = tuple(tuple(a.shape) for a in self.As)
        shapes_s = tuple(tuple(a.shape) for a in state.As)
        shape_sig = shapes_q + shapes_s

        if self._cache_cls.shape_sig != shape_sig:
            # Build fresh symbols for this combined network
            sym = GetSymbolFn()
            L = self.L

            # Physical indices shared between Q and S
            p = [sym() for _ in range(L)]
            # Bond indices for classifier (Q)
            b = [sym() for _ in range(L - 1)]
            # Bond indices for state (S)
            a = [sym() for _ in range(L - 1)]
            # Output class index
            c = sym()

            pieces: List[str] = []

            # Classifier tensors (MpsQsc)
            # Q0: (d, chi) -> (p0, b1)
            pieces.append(p[0] + b[0])
            # Q1..Q_{L-2}: (chi, d, chi) -> (b_i, p_i, b_{i+1})
            for i in range(1, L - 1):
                pieces.append(b[i - 1] + p[i] + b[i])
            # Q_{L-1}: (chi, d, 2) -> (b_{L-1}, p_{L-1}, c)
            pieces.append(b[L - 2] + p[L - 1] + c)

            # State tensors (MPState)
            # S0: (d, chi) -> (p0, a1)
            pieces.append(p[0] + a[0])
            # S1..S_{L-2}: (chi, d, chi) -> (a_i, p_i, a_{i+1})
            for i in range(1, L - 1):
                pieces.append(a[i - 1] + p[i] + a[i])
            # S_{L-1}: (chi, d) -> (a_{L-1}, p_{L-1})
            pieces.append(a[L - 2] + p[L - 1])

            eq = ",".join(pieces) + "->" + c

            path, _ = oe.contract_path(
                eq,
                *(self.As + state.As),
                optimize=self.optimize,
            )
            self._cache_cls.eq = eq
            self._cache_cls.path = path
            self._cache_cls.shape_sig = shape_sig

        if self._cache_cls.eq is None or self._cache_cls.path is None:
            raise ValueError("Classifier-state contraction equation or path not built.")

        return self._cache_cls.eq, self._cache_cls.path
    

    def contract_with_state(self, state: "MPState") -> torch.Tensor:
        """
        Contract this classification MPS with a given MPState.

        Mathematically:
          given W_{p1...pL, c} (MpsQsc) and ψ_{p1...pL} (MPState),
          return

              y_c = sum_{p1,...,pL} W_{p1...pL, c} * ψ_{p1...pL}

        Parameters
        ----------
        state : MPState
            The MPS quantum state to contract with.

        Returns
        -------
        torch.Tensor
            1D tensor of shape (2,), the two class scores.
        """
        # 1. Check compatibility (L, d, chi)
        if self.L != state.L:
            raise ValueError(f"Length mismatch: MpsQsc.L={self.L}, MPState.L={state.L}")
        if self.d != state.d:
            raise ValueError(f"Physical dim mismatch: MpsQsc.d={self.d}, MPState.d={state.d}")

        # 2. Make sure we're on the same device/dtype
        if getattr(state, "device", self.device) != self.device or getattr(state, "dtype", self.dtype) != self.dtype:
            state = state.to(device=self.device, dtype=self.dtype)

        # 3. Build (or reuse) equation + path, then contract
        eq, path = self._build_state_path(state)
        result = oe.contract(eq, *(self.As + state.As), optimize=path)

        # Should be shape (2,)
        return result
    
    def parameters(self) -> list[torch.Tensor]:
        """
        Return trainable tensors so you can do:
            opt = torch.optim.Adam(qsc.parameters(), lr=...)
        """
        return self.As

    def zero_grad(self) -> None:
        """Zero gradients of cores."""
        for A in self.As:
            if A.grad is not None:
                A.grad.zero_()

    def train(self, mode: bool = True):
        """
        No-op training toggle for API compatibility.
        (Useful if your loops call .train()/.eval().)
        """
        self.training = bool(mode)
        return self

    def eval(self):
        """No-op eval toggle (mirrors nn.Module)."""
        return self.train(False)
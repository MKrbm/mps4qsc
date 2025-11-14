from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import math
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from ..opt_einsum_utils import GetSymbolFn, _EqAndPathCache

class MPState:
    """
    Standard MPS (state, no classifier leg).

    Shapes (L sites):
      As[0]     : (d, chi)
      As[1..L-2]: (chi, d, chi)
      As[L-1]   : (chi, d)

    contract() returns a tensor of shape (d, d, ..., d) (L copies).

    Contraction uses opt_einsum; the equation + path are cached after the first call
    and reused thereafter.
    """
    def __init__(
        self,
        L: int,
        chi: int,
        d: int,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        init: str = "random",                  # mirrors your MpsQsc signature
        seed: Optional[int] = None,
        optimize: str = "random-greedy"        # or "greedy"
    ):
        if L < 2:
            raise ValueError("MPState requires L >= 2.")
        self.L = L
        self.chi = chi
        self.d = d
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.optimize = optimize

        self.As: List[torch.Tensor] = []
        self.symbols: List[List[str]] = []
        self._cache = _EqAndPathCache()
        self._norm_cache = _EqAndPathCache()
        self._ip_cache = _EqAndPathCache()
        self._sym = GetSymbolFn()

        if As is not None:
            self.set_As([a.to(self.device, self.dtype) for a in As])
        else:
            self.initialize(init=init, seed=seed)

    # ---------- Initialization ----------
    def initialize(self, init: str = "random", seed: Optional[int] = None):
        """
        Initialize tensors.
        Currently implements 'random' to match your MpsQsc; 'stacked_identity' is
        left explicit to avoid silent misconfigurations.
        """
        if seed is not None:
            torch.manual_seed(seed)

        if init not in {"random", "stacked_identity"}:
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
            # Last (right boundary)
            AL = torch.randn(chi, d, device=self.device, dtype=self.dtype) / math.sqrt(chi)
            As.append(AL)
        else:
            raise NotImplementedError(
                "MPState.initialize(init='stacked_identity') is not implemented in this version "
                "(mirrors your current MpsQsc)."
            )

        self.set_As(As)
        self._cache.invalidate()

    def to(self, device: Optional[torch.device | str] = None, dtype: Optional[torch.dtype] = None):
        """Move parameters to device/dtype."""
        if device is None and dtype is None:
            return self
        new_device = torch.device(device) if device is not None else self.device
        new_dtype = dtype if dtype is not None else self.dtype
        self.set_As([A.to(new_device, new_dtype) for A in self.As])
        self.device, self.dtype = new_device, new_dtype
        self._cache.invalidate()
        return self

    def set_As(self, As: Sequence[torch.Tensor]):
        """Replace tensors and rebuild symbol lists; reset cached path."""
        _validate_shapes_mpstate(self.L, self.d, self.chi, As)
        self.As = [A.to(self.device, self.dtype) for A in As]
        self._create_symbols()
        self._cache.invalidate()
    
    def _create_symbols(self):
        L = self.L
        sym = self._sym
        sym.reset()
        self.p_symbols = [sym() for _ in range(L)]
        self.b_symbols = [sym() for _ in range(L - 1)]
        self.symbols = [[self.p_symbols[0], self.b_symbols[0]]]
        for i in range(1, L - 1):
            self.symbols.append([self.b_symbols[i - 1], self.p_symbols[i], self.b_symbols[i]])
        self.symbols.append([self.b_symbols[L - 2], self.p_symbols[L - 1]])

    # ---------- Contraction ----------
    def _build_equation(self) -> str:
        """
        A0(p0,b1), A1(b1,p1,b2), ..., AL-2(bL-2,pL-2,bL-1), AL-1(bL-1,pL-1) -> p0 p1 ... pL-1
        """
        pieces = ["".join(self.symbols[i]) for i in range(self.L)]
        eq = ",".join(pieces) + "->" + "".join(self.p_symbols)
        return eq

    def _build_path(self) -> Tuple[str, oe.Path]:
        """
        Build (and cache) the opt_einsum path for the current shapes.
        """
        shapes = tuple(tuple(a.shape) for a in self.As)
        if self._cache.shape_sig != shapes:
            eq = self._build_equation()
            path, _ = oe.contract_path(eq, *self.As, optimize=self.optimize)
            self._cache.eq = eq
            self._cache.path = path
            self._cache.shape_sig = shapes
        if self._cache.eq is None or self._cache.path is None:
            raise ValueError("Equation or path is not built.")
        return self._cache.eq, self._cache.path

    def state_vector(self) -> torch.Tensor:
        """
        Contract the whole MPS to yield a tensor with shape (d,)*L.

        NOTE: This materializes O(d^L) entries.
        """
        self._build_path()
        return oe.contract(self._cache.eq, *self.As, optimize=self._cache.path)

    def _build_overlap_path(self, other: "MPState") -> Tuple[str, oe.Path]:
        """
        Build (and cache) the opt_einsum equation + path for the overlap ⟨other|self⟩.
        We share the physical indices p_i and keep separate bond indices (ak for self, ab for other).
        Output is a scalar.

        Self (ket):  A0(p0, ak1), A1(ak1, p1, ak2), ..., AL-1(ak_{L-1}, p_{L-1})
        Other (bra): B0(p0, ab1), B1(ab1, p1, ab2), ..., BL-1(ab_{L-1}, p_{L-1})

        Tensors will be passed to `contract` in this order: [self.As..., conj(other.As)...].
        """
        # --- shape key
        shapes_self = tuple(tuple(a.shape) for a in self.As)
        shapes_other = tuple(tuple(a.shape) for a in other.As)
        shape_sig = shapes_self + shapes_other

        if self._ip_cache.shape_sig != shape_sig:
            L = self.L
            sym = GetSymbolFn()

            # Shared physical indices; distinct bond indices
            p  = [sym() for _ in range(L)]
            ak = [sym() for _ in range(L - 1)]  # bonds for self (ket)
            ab = [sym() for _ in range(L - 1)]  # bonds for other (bra)

            pieces: List[str] = []
            # self / ket
            pieces.append(p[0] + ak[0])
            for i in range(1, L - 1):
                pieces.append(ak[i - 1] + p[i] + ak[i])
            pieces.append(ak[L - 2] + p[L - 1])
            # other / bra
            pieces.append(p[0] + ab[0])
            for i in range(1, L - 1):
                pieces.append(ab[i - 1] + p[i] + ab[i])
            pieces.append(ab[L - 2] + p[L - 1])

            eq = ",".join(pieces) + "->"  # scalar

            # Build once—a path that works for [self.As..., other.As...]
            # (conjugation doesn't change shapes, so the path is the same)
            path, _ = oe.contract_path(eq, *(self.As + other.As), optimize=self.optimize)
            self._ip_cache.eq = eq
            self._ip_cache.path = path
            self._ip_cache.shape_sig = shape_sig

        if self._ip_cache.eq is None or self._ip_cache.path is None:
            raise ValueError("Equation or path for inner_product() not built.")
        return self._ip_cache.eq, self._ip_cache.path
    

    def inner_product(self, other: "MPState") -> torch.Tensor:
        """
        Return the scalar overlap ⟨other|self⟩ as a 0-D tensor.
        Validates (L, d, chi). Uses a cached opt_einsum path.
        """
        if self.L != other.L:
            raise ValueError(f"Length mismatch: self.L={self.L}, other.L={other.L}")
        if self.d != other.d:
            raise ValueError(f"Physical dimension mismatch: self.d={self.d}, other.d={other.d}")
        if self.chi != other.chi:
            raise ValueError(f"Bond dimension mismatch: self.chi={self.chi}, other.chi={other.chi}")

        # Build (or reuse) equation and path
        eq, path = self._build_overlap_path(other)

        # Tensors: ket followed by bra (conjugated)
        ket = self.As
        bra = [torch.conj(B.to(self.device, self.dtype)) for B in other.As]

        return oe.contract(eq, *(ket + bra), optimize=path)


    def norm(self, squared: bool = False) -> torch.Tensor:
        """
        Return ||psi|| (default) or ||psi||^2 (if squared=True),
        computed via inner_product(self, self).
        """
        ip = self.inner_product(self)  # ⟨self|self⟩
        ip_real = ip.real if torch.is_complex(ip) else ip
        if squared:
            return ip_real
        # numerical safety
        return torch.sqrt(torch.clamp(ip_real, min=0))


    def normalize(self, inplace: bool = True) -> "MPState":
        """
        Scale the MPS so that ||psi|| = 1 by distributing the scaling evenly across all sites.

        If the state has norm n, we multiply each core by s = n^(-1/L), so the full
        state (and thus its vector) scales by s^L = 1/n.

        Returns
        -------
        MPState
            self if inplace=True, otherwise a new MPState instance with scaled cores.
        """
        nrm = self.norm(squared=False)
        if not torch.isfinite(nrm):
            raise ValueError("Norm is not finite; cannot normalize.")
        if float(nrm) == 0.0:
            raise ValueError("Zero-norm state; cannot normalize.")

        # uniform per-site scale
        s = torch.pow(nrm, -1.0 / float(self.L))
        s = s.to(device=self.device, dtype=self.dtype)

        if inplace:
            self.As = [A * s for A in self.As]
            # values changed but shapes unchanged, so cached paths remain valid
            return self

        new_As = [A * s for A in self.As]
        return MPState(
            L=self.L, chi=self.chi, d=self.d,
            As=new_As, device=self.device, dtype=self.dtype,
            optimize=self.optimize,
        )





def _validate_shapes_mpstate(L: int, d: int, chi: int, As: Sequence[torch.Tensor]):
    if len(As) != L:
        raise ValueError(f"As must have length L={L}, got {len(As)}.")
    if L < 2:
        raise ValueError("MPState requires L >= 2 (first, interior..., last).")
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
    if tuple(As[L - 1].shape) != (chi, d):
        raise ValueError(f"As[L-1] must have shape (chi, d)=({chi},{d}), "
                         f"got {tuple(As[L-1].shape)}.")
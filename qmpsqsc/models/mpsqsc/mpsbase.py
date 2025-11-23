from __future__ import annotations
from typing import Optional, Sequence, Tuple, List, Union
from abc import ABC, abstractmethod
import math
import torch
import os
from pathlib import Path
from typing import Dict, Any, Self
import opt_einsum as oe  # type: ignore[import-untyped]
from ..opt_einsum_utils import GetSymbolFn, _EqAndPathCache




class MPSBase:
    """
    Unified MPS class.

    Shapes (L sites; out_dim=C):
      As[0]     : (d, chi)
      As[1..L-2]: (chi, d, chi)
      As[L-1]   : (chi, d)          if C == 1
                  (chi, d, C)       if C > 1

    full_vector():
      returns a tensor with shape (d,)*L                if C == 1
      returns a tensor with shape (d,)*L + (C,)         if C > 1

    overlap(other, conjugate_self=True):
      - state vs state (C=self=1, other=1): scalar
      - classifier vs state: vector of shape (C,)      (works in either order)
      - classifier vs classifier: matrix of shape (C_self, C_other)

    Notes
    -----
    * Equations and opt_einsum paths are cached. There is no `_create_symbols`.
    * For `norm()` with C>1, we use the Frobenius norm of the full tensor:
      ||W||^2 = sum_c ⟨W_c | W_c⟩ = trace( overlap(self, self) ).
    """

    def __init__(
        self,
        L: int,
        d: int,
        out_dim: int = 1,
        chi: int | Sequence[int] | None = None,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
        init: str = "random",
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
    ):
        if L < 2:
            raise ValueError("MPS requires L >= 2.")
        if out_dim < 1:
            raise ValueError("out_dim must be >= 1.")

        self.L = L
        self.d = d
        self.out_dim = out_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.optimize = optimize

        # Caches
        self._full_cache = _EqAndPathCache()   # for full_vector()
        self._ovlp_cache = _EqAndPathCache()   # for overlap(...)

        # Per-bond bond dimensions (None means "uniform chi=self.chi")
        self._chi_bonds: List[int] = []

        # Determine chi / chi_bonds depending on whether As are provided
        if As is None:
            # We must be given chi (int or list) if we don't get As
            if chi is None:
                raise ValueError("Either `chi` (int or list) or `As` must be provided.")
            if isinstance(chi, int):
                chi_max = chi
                chi_bonds: List[int] = [chi] * (L - 1)  # uniform
            else:
                chi_bonds = list(chi)
                if len(chi_bonds) != L - 1:
                    raise ValueError(
                        f"`chi` as a sequence must have length L-1={L-1}, "
                        f"got {len(chi_bonds)}."
                    )
                chi_max = max(chi_bonds) if chi_bonds else 1

            self.chi_max = chi_max
            self._chi_bonds = chi_bonds
            # self.As: List[torch.Tensor] = []
            self.initialize(init=init, seed=seed)

        else:
            # As are given; chi is optional and acts as a consistency bound.
            # Infer bond dims from As
            bonds_from_As = _infer_bond_dims_from_As(L, d, out_dim, As)
            chi_max_from_As = max(bonds_from_As) if bonds_from_As else 1

            if chi is None:
                chi_max = chi_max_from_As
            elif isinstance(chi, int):
                chi_max = chi
                if chi_max < chi_max_from_As:
                    raise ValueError(
                        f"Provided chi={chi_max} is smaller than the maximum bond "
                        f"dimension implied by As ({chi_max_from_As})."
                    )
            else:
                chi_list = list(chi)
                if len(chi_list) != len(bonds_from_As):
                    raise ValueError(
                        f"Provided chi list has length {len(chi_list)}, but As implies "
                        f"{len(bonds_from_As)} bonds."
                    )
                if chi_list != bonds_from_As:
                    raise ValueError(
                        "Provided chi list does not match bond dimensions inferred from As: "
                        f"{chi_list} vs {bonds_from_As}."
                    )
                chi_max = max(chi_list) if chi_list else 1

            self.chi_max = chi_max
            self._chi_bonds = bonds_from_As
            # self.As: List[torch.Tensor] = []
            # set_As will re-validate and re-derive _chi_bonds (same result)
            self.set_As([a.to(self.device, self.dtype) for a in As])

    def _random_initialize(self, seed: Optional[int] = None) -> List[torch.Tensor]:
        """Initialize the MPS with random fan-in scaled initialization."""
        # if seed is not None:
        #     torch.manual_seed(seed)

        d, L, C = self.d, self.L, self.out_dim
        chi_bonds = self._chi_bonds
        As: List[torch.Tensor] = []

        if chi_bonds is None:
            # Uniform bond dimension chi = self.chi (original behavior)
            chi = self.chi_max
            # First
            A0 = torch.randn(d, chi, device=self.device, dtype=self.dtype) / math.sqrt(chi)
            As.append(A0)

            # Interiors
            for _ in range(1, L - 1):
                A = (
                    torch.randn(chi, d, chi, device=self.device, dtype=self.dtype)
                    / math.sqrt(chi * d)
                )
                As.append(A)

            # Last
            if C == 1:
                AL = torch.randn(chi, d, device=self.device, dtype=self.dtype) / math.sqrt(chi)
            else:
                AL = (
                    torch.randn(chi, d, C, device=self.device, dtype=self.dtype)
                    / math.sqrt(chi * d)
                )
            As.append(AL)
        else:
            # Site-dependent bond dimensions
            if len(chi_bonds) != L - 1:
                raise ValueError(
                    f"Internal _chi_bonds must have length L-1={L-1}, "
                    f"got {len(chi_bonds)}."
                )

            # First
            chi0 = chi_bonds[0]
            A0 = torch.randn(d, chi0, device=self.device, dtype=self.dtype) / math.sqrt(chi0)
            As.append(A0)

            # Interiors
            for i in range(1, L - 1):
                chi_left = chi_bonds[i - 1]
                chi_right = chi_bonds[i]
                A = (
                    torch.randn(chi_left, d, chi_right, device=self.device, dtype=self.dtype)
                    / math.sqrt(chi_left * d)
                )
                As.append(A)

            # Last
            chi_last = chi_bonds[L - 2]
            if C == 1:
                AL = torch.randn(chi_last, d, device=self.device, dtype=self.dtype) / math.sqrt(chi_last)
            else:
                AL = (
                    torch.randn(chi_last, d, C, device=self.device, dtype=self.dtype)
                    / math.sqrt(chi_last * d)
                )
            As.append(AL)

        return As

    
    def _initialize(self, init: str = "random", seed: Optional[int] = None) -> List[torch.Tensor]:
        """Initialize the MPS with a given initialization method."""
        raise NotImplementedError("Subclasses must implement this method.")

    def set_requires_grad(self, requires_grad: bool = True):
        for A in self.As:
            A.requires_grad_(requires_grad)
        self._invalidate_caches()

    def initialize(self, init: str = "random", seed: Optional[int] = None):
        """Random fan-in scaled initialization."""
        if init not in {"random"}:
            As = self._initialize(init=init, seed=seed)
        else:
            As = self._random_initialize(seed=seed)
        self.set_As(As)
        self._full_cache.invalidate()

    def set_As(self, As: Sequence[torch.Tensor]):
        """Replace cores, canonicalizing the last-site rank for out_dim==1; reset caches."""
        # Validate against current chi bound
        _validate_shapes_base(self.L, self.d, self.chi_max, self.out_dim, As)

        new_As: List[torch.Tensor] = []
        for i, A in enumerate(As):
            T = A.to(self.device, self.dtype).detach().clone()
            new_As.append(T)

        # Canonicalize last core: for out_dim==1, ensure shape (chi, d) (squeeze trailing 1 if present)
        if self.out_dim == 1 and new_As[-1].ndim == 3 and new_As[-1].shape[2] == 1:
            new_As[-1] = new_As[-1][..., 0]  # (chi, d)

        self.As = [A.detach().clone() for A in new_As]

        # Update per-bond chi list from the new cores
        self._chi_bonds = _infer_bond_dims_from_As(self.L, self.d, self.out_dim, self.As)

        self._invalidate_caches()


    def _invalidate_caches(self):
        self._full_cache.invalidate()
        self._ovlp_cache.invalidate()

    # ---------- Full contraction (materialize tensor) ----------
    def _build_full_equation(self) -> tuple[str, dict[str, List[str] | str], GetSymbolFn]:
        """
        Build einsum eq for:
           A0(p0,b1), A1(b1,p1,b2), ..., AL-2(bL-2,pL-2,bL-1), AL-1(bL-1,pL-1[,c])
        -> p0 p1 ... pL-1 [c]
        """
        symfn = GetSymbolFn()
        L = self.L

        p = [symfn() for _ in range(L)]
        b = [symfn() for _ in range(L - 1)]
        c = symfn() if self.out_dim > 1 else ""

        pieces: List[str] = []
        pieces.append(p[0] + b[0])
        for i in range(1, L - 1):
            pieces.append(b[i - 1] + p[i] + b[i])
        if self.out_dim > 1:
            pieces.append(b[L - 2] + p[L - 1] + c)  # (chi, d, C)
            eq = ",".join(pieces) + "->" + "".join(p) + c
        else:
            pieces.append(b[L - 2] + p[L - 1])      # (chi, d)
            eq = ",".join(pieces) + "->" + "".join(p)
        syms: Dict[str, List[str] | str] = {}
        syms["p"] = p
        syms["b"] = b
        if c != "":
            syms["c"] = c
        return eq, syms, symfn

    def _build_fv_path(self) -> Tuple[str, oe.Path]:
        shapes = tuple(tuple(a.shape) for a in self.As)
        if self._full_cache.shape_sig != shapes:
            eq, _, _ = self._build_full_equation()
            path, _ = oe.contract_path(eq, *self.As, optimize=self.optimize)
            self._full_cache.eq = eq
            self._full_cache.path = path
            self._full_cache.shape_sig = shapes
        if self._full_cache.eq is None or self._full_cache.path is None:
            raise ValueError("Equation or path is not built.")
        return self._full_cache.eq, self._full_cache.path

    def full_vector(self) -> torch.Tensor:
        """
        Contract the whole MPS to yield:
          shape (d,)*L                 if out_dim == 1
          shape (d,)*L + (out_dim,)    if out_dim > 1

        NOTE: Exponentially large in L.
        """
        eq, path = self._build_fv_path()
        return oe.contract(eq, *self.As, optimize=path)

    # Back-compat alias for code that still calls .state_vector()
    def state_vector(self) -> torch.Tensor:
        return self.full_vector()

    # ---------- Overlap ----------
    def _build_overlap_equation(self, other: "MPSBase") -> str:
        """
        Build eq for overlap(self, other). We do NOT bake conjugation into
        the equation; conjugation is applied to tensors before contracting.

          Self (A): A0(p0,a1), A1(a1,p1,a2), ..., AL-1(a_{L-1},p_{L-1}[,cA])
          Other(B): B0(p0,b1), B1(b1,p1,b2), ..., BL-1(b_{L-1},p_{L-1}[,cB])

        Output indices: [cA][cB] (in that order), only if the corresponding out_dim > 1.
        """
        if self.L != other.L:
            raise ValueError(f"Length mismatch: {self.L} vs {other.L}")
        if self.d != other.d:
            raise ValueError(f"Physical dim mismatch: {self.d} vs {other.d}")

        sym = GetSymbolFn()
        L = self.L

        p = [sym() for _ in range(L)]
        a = [sym() for _ in range(L - 1)]
        b = [sym() for _ in range(L - 1)]
        cA = sym() if self.out_dim > 1 else None
        cB = sym() if other.out_dim > 1 else None

        pieces: List[str] = []
        # Self
        pieces.append(p[0] + a[0])
        for i in range(1, L - 1):
            pieces.append(a[i - 1] + p[i] + a[i])
        pieces.append(a[L - 2] + p[L - 1] + (cA or ""))

        # Other
        pieces.append(p[0] + b[0])
        for i in range(1, L - 1):
            pieces.append(b[i - 1] + p[i] + b[i])
        pieces.append(b[L - 2] + p[L - 1] + (cB or ""))

        out = ""
        if cA:
            out += cA
        if cB:
            out += cB
        eq = ",".join(pieces) + "->" + out
        return eq

    def _build_overlap_path(self, other: "MPSBase") -> Tuple[str, oe.Path]:
        # Shapes signature: self + other (conjugation doesn't affect shapes)
        shapes_self = tuple(tuple(a.shape) for a in self.As)
        shapes_other = tuple(tuple(a.shape) for a in other.As)
        shape_sig = shapes_self + shapes_other

        if self._ovlp_cache.shape_sig is None:
            eq = self._build_overlap_equation(other)
            # Build path for the concatenated argument list (self first, then other)
            path, _ = oe.contract_path(eq, *(self.As + other.As), optimize=self.optimize)
            self._ovlp_cache.eq = eq
            self._ovlp_cache.path = path
            self._ovlp_cache.shape_sig = shape_sig

        if self._ovlp_cache.eq is None or self._ovlp_cache.path is None:
            raise ValueError("Overlap equation/path not built.")
        return self._ovlp_cache.eq, self._ovlp_cache.path

    def overlap(self, other: "MPSBase", conjugate_self: bool = True) -> torch.Tensor:
        """
        ⟨self|other⟩ by default (i.e., conjugate_self=True).
        Returns:
          - scalar                         if self.out_dim==other.out_dim==1
          - (C,)                           if one has out_dim==C>1 and the other has 1
          - (C_self, C_other)              if both have out_dim>1

        Set `conjugate_self=False` to compute a raw contraction without taking complex conjugate
        of `self` (useful for classifier scores).
        """
        # Device/dtype alignment: do NOT mutate the inputs.
        lhs = [A.to(self.device, self.dtype) for A in self.As]
        rhs = [B.to(self.device, self.dtype) for B in other.As]

        if conjugate_self:
            lhs = [torch.conj(A) for A in lhs]

        eq, path = self._build_overlap_path(other)
        return oe.contract(eq, *(lhs + rhs), optimize=path)

    # ---------- Norm & normalization ----------
    def norm(self, squared: bool = False) -> torch.Tensor:
        """
        For out_dim==1: ||ψ|| (or ||ψ||^2).
        For out_dim>1 : Frobenius norm of the full tensor (sum over all class legs):
                        ||W||^2 = trace( overlap(self, self) ).
        """
        ov = self.overlap(self, conjugate_self=True)  # scalar or (C,C) Gram
        if ov.ndim == 0:
            n2 = ov.real if torch.is_complex(ov) else ov
        else:
            # Gram matrix; trace is sum of class-channel norms
            ov_real = ov.real if torch.is_complex(ov) else ov
            n2 = torch.trace(ov_real)
        if squared:
            return n2
        return torch.sqrt(torch.clamp(n2, min=0))

    def normalize(self, inplace: bool = True, norm: Optional[float | torch.Tensor] = None) -> Self:
        """
        Scale cores so that `norm() == 1` by multiplying each core by s = n^(-1/L).
        Works for both C==1 and C>1.
        If norm is provided, rescale the cores based on the provided norm.
        """
        nrm = self.norm(squared=False) if norm is None else norm
        if not isinstance(nrm, torch.Tensor):
            nrm = torch.tensor(nrm, device=self.device, dtype=self.dtype)
        if not torch.isfinite(nrm):
            raise ValueError("Norm is not finite; cannot normalize.")
        if float(nrm) == 0.0:
            raise ValueError("Zero-norm state; cannot normalize.")

        s = torch.pow(nrm, -1.0 / float(self.L))
        scaled = [A * s for A in self.As]

        if inplace:
            # Keep shapes identical; cached paths remain valid (shapes unchanged).
            for i in range(len(self.As)):
                self.As[i].data[:].mul_(s)
            return self

        return self._clone_with_As(scaled)

    # ---------- Training helpers ----------
    def parameters(self) -> list[torch.Tensor]:
        return [A for A in self.As if A.requires_grad]

    def zero_grad(self) -> None:
        for A in self.As:
            if A.grad is not None:
                A.grad.zero_()

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    @abstractmethod
    def _clone_with_As(self, As_new: Sequence[torch.Tensor], new_chi: int | None = None) -> Self:
        pass

    # ---------- Optional: bond-dimension truncation (TT-SVD) ----------
    def truncate_bond_dimension(self, new_chi: int) -> Self:
        """
        Reduce the internal bond dimension to at most `new_chi`
        via sequential SVD compression (TT-SVD). Supports both C==1 and C>1.
        Returns the effective max bond dimension `chi_eff <= new_chi`.
        """
        # if new_chi >= self.chi:
        #     return self._clone_with_As(self.As)
        # if new_chi < 1:
        #     raise ValueError("new_chi must be >= 1")

        def _svd(mat: torch.Tensor):
            try:
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            except AttributeError:
                U, S, V = torch.svd(mat)
                Vh = V.mH if hasattr(V, "mH") else V.t()
            return U, S, Vh

        with torch.no_grad():
            As = [A.detach().clone() for A in self.As]
            L, d, C = self.L, self.d, self.out_dim

            Bs: list[torch.Tensor] = []

            # Site 0: (d, chi0)
            A0 = As[0]
            mat = A0  # (d, chi0)
            U, S, Vh = _svd(mat)
            r0 = min(new_chi, S.shape[0])
            U = U[:, :r0]            # (d, r0)
            S = S[:r0]               # (r0,)
            Vh = Vh[:r0, :]          # (r0, chi0)
            Bs.append(U)             # B0: (d, r0)
            M = (S.unsqueeze(1) * Vh)  # (r0, chi0)

            # Prepare the "core" after pushing R to site 1
            if L == 2:
                # Last site directly
                last = As[1]  # (chi, d) or (chi, d, C)
                if last.ndim == 2:  # (chi, d)
                    core = torch.einsum("ab,bd->ad", M, last)      # (r0, d)
                else:  # (chi, d, C)
                    core = torch.einsum("ab,bdc->adc", M, last)    # (r0, d, C)
            else:
                nxt = As[1]  # (chi, d, chi)
                core = torch.einsum("ab,bdc->adc", M, nxt)         # (r0, d, chi)

                for site in range(1, L - 1):
                    chi_left, d_site, chi_right_or_C = core.shape if core.ndim == 3 else (core.shape[0], core.shape[1], 1)
                    # Flatten left legs
                    mat = core.reshape(chi_left * d_site, chi_right_or_C)
                    U, S, Vh = _svd(mat)
                    r = min(new_chi, S.shape[0])
                    U = U[:, :r]
                    S = S[:r]
                    Vh = Vh[:r, :]

                    B = U.reshape(chi_left, d_site, r)  # (chi_left, d, r)
                    Bs.append(B)
                    M = (S.unsqueeze(1) * Vh)           # (r, chi_right_or_C)

                    if site + 1 < L:
                        next_core = As[site + 1]
                        if next_core.ndim == 3:
                            core = torch.einsum("ab,bdc->adc", M, next_core)
                        else:
                            # last site and C==1
                            core = torch.einsum("ab,bd->ad", M, next_core)

                # 'core' now is last site
                Bs.append(core)

            # Determine per-bond ranks
            bond_dims: list[int] = []
            bond_dims.append(Bs[0].shape[1])  # between site 0 and 1
            for i in range(1, L - 1):
                # Bs[i]: (r_{i-1}, d, r_i)
                bond_dims.append(Bs[i].shape[2])

            chi_eff = max(bond_dims) if bond_dims else 1

            # Re-embed into uniform chi_eff
            new_As: list[torch.Tensor] = []

            # First: (d, chi_eff)
            A0_new = torch.zeros(d, chi_eff, device=self.device, dtype=self.dtype)
            r0 = bond_dims[0]
            A0_new[:, :r0] = Bs[0]
            new_As.append(A0_new)

            # Interiors: (chi_eff, d, chi_eff)
            for i in range(1, L - 1):
                left_r = bond_dims[i - 1]
                right_r = bond_dims[i]
                Bi = Bs[i]  # (left_r, d, right_r)
                Ai_new = torch.zeros(chi_eff, d, chi_eff, device=self.device, dtype=self.dtype)
                Ai_new[:left_r, :, :right_r] = Bi
                new_As.append(Ai_new)

            # Last
            last_block = Bs[L - 1]
            if self.out_dim == 1:
                # (r_{L-2}, d)
                A_last_new = torch.zeros(chi_eff, d, device=self.device, dtype=self.dtype)
                A_last_new[:bond_dims[-1], :] = last_block
            else:
                # (r_{L-2}, d, C)
                A_last_new = torch.zeros(chi_eff, d, C, device=self.device, dtype=self.dtype)
                A_last_new[:bond_dims[-1], :, :] = last_block
            new_As.append(A_last_new)

        return self._clone_with_As(new_As, new_chi=chi_eff)

    def canonicalize(
        self,
        inplace: bool = False,
        normalize: bool = False,
        truncate: bool = False,
    ) -> Self:
        """
        Left-canonicalize the MPS (0 -> L-1) via a QR sweep.

        - If truncate=False:
            We keep the configured bond dimensions and only change the cores:
              * If `self._chi_bonds` is None:
                    A[0]     : (d, chi)
                    A[1..L-2]: (chi, d, chi)
                    A[L-1]   : (chi, d[, C])
                (zeros are used if the effective rank is smaller than chi.)
              * If `self._chi_bonds` is a list [chi_0, ..., chi_{L-2}]:
                    A[0]     : (d, chi_0)
                    A[1]     : (chi_0, d, chi_1)
                    ...
                    A[L-1]   : (chi_{L-2}, d[, C])

        - If truncate=True:
            We detect numerical rank deficiencies at each bond and
            *shrink* the bond dimensions to the effective ranks. Shapes become
            variable:
                A[0]     : (d, r0)
                A[1]     : (r0, d, r1)
                ...
                A[L-1]   : (r_{L-2}, d[, C])
            while remaining left-canonical.

        Parameters
        ----------
        inplace : bool, default False
            If True, modify this instance; otherwise return a new canonicalized copy.
        normalize : bool, default False
            If True, normalize the last tensor to have unit Frobenius norm.
        truncate : bool, default False
            If True, use an effective-rank criterion on R to shrink bonds.

        Returns
        -------
        MPSBase
        """
        L, d, C = self.L, self.d, self.out_dim
        device, dtype = self.device, self.dtype

        def _qr_truncated(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
            """
            QR + numerical rank detection from diag(R).

            Returns
            -------
            Q_eff : (m, r_eff)
            R_eff : (r_eff, n)
            r_eff : int
            """
            Q, R = torch.linalg.qr(M, mode="reduced")
            diag = torch.diagonal(R)          # (k,)
            abs_diag = diag.abs()
            max_diag = abs_diag.max()

            # Heuristic tolerance: similar to LAPACK-style
            if max_diag == 0:
                # Matrix is numerically all zeros; keep rank 1 to avoid degenerate shapes.
                r_eff = 1
            else:
                eps = torch.finfo(M.dtype).eps
                tol = max(M.shape) * eps * max_diag
                r_eff = int((abs_diag > tol).sum().item())
                if r_eff == 0:
                    r_eff = 1

            Q_eff = Q[:, :r_eff]
            R_eff = R[:r_eff, :]
            return Q_eff, R_eff, r_eff

        # Target per-bond χ for the non-truncate path:
        #   - if self._chi_bonds is set: use those
        #   - otherwise: assume uniform chi = self.chi
        chi_bonds_attr = getattr(self, "_chi_bonds", None)
        if chi_bonds_attr is not None:
            if len(chi_bonds_attr) != L - 1:
                raise ValueError(
                    f"Internal _chi_bonds must have length L-1={L-1}, "
                    f"got {len(chi_bonds_attr)}."
                )
            chi_bonds_target: list[int] = list(chi_bonds_attr)
        else:
            chi_bonds_target = [self.chi_max] * (L - 1)

        # Work on copies to preserve autograd flags later
        src_As = [A.to(device, dtype) for A in self.As]

        Bs: list[torch.Tensor] = [torch.zeros(1, 1, device=device, dtype=dtype) for _ in range(L)]  # type: ignore[assignment]

        with torch.no_grad():
            # ---- Site 0: A0 (d, chi_0) -> Q0(d, r0), push R(r0, chi_0) right ----
            A0 = src_As[0]
            if truncate:
                Q0, R, r_prev = _qr_truncated(A0)    # Q0: (d, r0)
                Bs[0] = Q0                           # (d, r0)
            else:
                Q0, R = torch.linalg.qr(A0, mode="reduced")  # Q0: (d, k)
                r_prev = Q0.shape[1]
                chi0_target = chi_bonds_target[0]
                if r_prev > chi0_target:
                    raise ValueError(
                        f"Effective rank r0={r_prev} exceeds target bond dim "
                        f"chi_0={chi0_target}."
                    )
                B0 = torch.zeros(d, chi0_target, device=device, dtype=dtype)
                B0[:, :r_prev] = Q0
                Bs[0] = B0

            # ---- Sweep interiors 1..L-2 ----
            for i in range(1, L - 1):
                Ai = src_As[i]   # (chi_{i-1}, d, chi_i) in general

                # Absorb incoming R on the left bond:
                #   R  : (r_prev, chi_{i-1})
                #   Ai : (chi_{i-1}, d, chi_i)
                left_absorbed = torch.einsum("ra,adc->rdc", R, Ai)  # (r_prev, d, chi_i)
                chi_right_src = Ai.shape[2]
                M = left_absorbed.reshape(r_prev * d, chi_right_src)

                if truncate:
                    Q, R, r = _qr_truncated(M)        # Q: (r_prev*d, r)
                    Bi = Q.reshape(r_prev, d, r)      # (r_prev, d, r)
                else:
                    Q, R = torch.linalg.qr(M, mode="reduced")
                    r = Q.shape[1]
                    chi_left_target = chi_bonds_target[i - 1]
                    chi_right_target = chi_bonds_target[i]
                    if r_prev > chi_left_target or r > chi_right_target:
                        raise ValueError(
                            f"Effective ranks (left={r_prev}, right={r}) exceed "
                            f"target bond dims (chi_left={chi_left_target}, "
                            f"chi_right={chi_right_target}) at site {i}."
                        )
                    Bi = torch.zeros(
                        chi_left_target, d, chi_right_target,
                        device=device, dtype=dtype
                    )
                    Bi[:r_prev, :, :r] = Q.reshape(r_prev, d, r)

                Bs[i] = Bi
                r_prev = r

            # ---- Last site: absorb final R on its left bond ----
            last = src_As[L - 1]
            if C == 1:
                # last: (chi_{L-2}, d)
                absorbed = torch.einsum("rb,bd->rd", R, last)       # (r_prev, d)
                if truncate:
                    BL = absorbed                                   # (r_prev, d)
                else:
                    chi_last_target = chi_bonds_target[L - 2]
                    if r_prev > chi_last_target:
                        raise ValueError(
                            f"Effective rank r_last={r_prev} exceeds target bond "
                            f"dim chi_{L-2}={chi_last_target}."
                        )
                    BL = torch.zeros(chi_last_target, d, device=device, dtype=dtype)
                    BL[:r_prev, :] = absorbed
            else:
                # last: (chi_{L-2}, d, C)
                absorbed = torch.einsum("rb,bdc->rdc", R, last)     # (r_prev, d, C)
                if truncate:
                    BL = absorbed                                   # (r_prev, d, C)
                else:
                    chi_last_target = chi_bonds_target[L - 2]
                    if r_prev > chi_last_target:
                        raise ValueError(
                            f"Effective rank r_last={r_prev} exceeds target bond "
                            f"dim chi_{L-2}={chi_last_target}."
                        )
                    BL = torch.zeros(
                        chi_last_target, d, C,
                        device=device, dtype=dtype
                    )
                    BL[:r_prev, :, :] = absorbed

            Bs[L - 1] = BL

            # Optional normalization of the last core
            if normalize:
                norm = torch.linalg.norm(Bs[L - 1])
                if norm > 0:
                    Bs[L - 1] /= norm

            # ---- Compute new per-bond χ from Bs ----
            new_bonds: list[int] = []
            # bond between site 0 and 1
            new_bonds.append(Bs[0].shape[1])
            # bonds between i and i+1
            for i in range(1, L - 1):
                Bi = Bs[i]
                if Bi.ndim != 3:
                    raise ValueError(
                        f"Bs[{i}] must be rank-3 with shape (chi_left, d, chi_right), "
                        f"got {tuple(Bi.shape)}."
                    )
                new_bonds.append(Bi.shape[2])

            new_chi = max(new_bonds) if new_bonds else 1

            # Validate shapes with the new chi bound
            _validate_shapes_base(self.L, self.d, new_chi, self.out_dim, Bs)


        if inplace:
            # Mutate this instance
            for i, B in enumerate(Bs):
                requires_grad = self.As[i].requires_grad
                self.As[i] = B.data[:]
                self.As[i].requires_grad_(requires_grad)
            self.chi_max = new_chi
            # Update per-bond chi list if present
            self._chi_bonds = new_bonds
            self._invalidate_caches()
            return self

        # Otherwise, construct a new instance
        out = self._clone_with_As(Bs, new_chi=new_chi)
        out.chi_max = new_chi
        if hasattr(out, "_chi_bonds"):
            out._chi_bonds = new_bonds
        out._invalidate_caches()
        return out


    def save_to(self, path: str | os.PathLike, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save this MPS to a file.

        Notes
        -----
        * Tensors are saved on CPU for portability; dtype is preserved.
        * On load you can choose a target device/dtype.

        Parameters
        ----------
        path : str | os.PathLike
            Destination filename (e.g., "model.pth").
        metadata : dict, optional
            Extra user-defined info to store alongside the model.
        """
        payload: dict[str, Any] = {
            "format": "mps_v2",
            "class_name": self.__class__.__name__,  # "MPState", "MpsQsc", or "MPSBase"
            "L": self.L,
            "d": self.d,
            "out_dim": getattr(self, "out_dim", 1),
            "optimize": self.optimize,
            "requires_grad": getattr(self, "requires_grad", False),
            "training": getattr(self, "training", False),
            # Store cores on CPU for portability; dtype is preserved.
            "As": [A.detach().to("cpu") for A in self.As],
            "metadata": dict(metadata) if metadata is not None else {},
        }
        torch.save(payload, path)
    
    def copy(self) -> Self:
        return self._clone_with_As(self.As)
    
    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Self:
        if device is None and dtype is None:
            raise ValueError("At least one of 'device' or 'dtype' must be provided to .to().")
        return self._clone_with_As([A.to(device=device, dtype=dtype) for A in self.As])
    
    def __mul__(self, scalar: float | complex | torch.Tensor) -> Self:
        """
        Scalar multiplication: MPS * scalar.
        Returns a new instance, does not modify in-place.
        """
        if not isinstance(scalar, (int, float, complex, torch.Tensor)):
            return NotImplemented
        if not isinstance(scalar, torch.Tensor):
            scalar = torch.tensor(scalar, device=self.device, dtype=self.dtype)
        phase = scalar / torch.abs(scalar)
        # Normalize a copy (returns new instance), then multiply first core by phase
        obj = self._clone_with_As(self.As)
        obj = obj.normalize(inplace=False, norm=1 / torch.abs(scalar))
        As_new = list(obj.As)
        As_new[0] = As_new[0] * phase
        return self._clone_with_As(As_new)

    def __rmul__(self, scalar: float | complex | torch.Tensor) -> Self:
        """
        Scalar multiplication: scalar * MPS
        """
        return self.__mul__(scalar)

def _validate_shapes_base(
    L: int,
    d: int,
    chi: int,                 # maximum allowed bond dimension
    out_dim: int,
    As: Sequence[torch.Tensor],
) -> None:
    """
    Validate an MPS/MPO-like list of tensors `As` with relaxed bond checks.

    Requirements enforced:
      • len(As) == L and L >= 2
      • Physical dimension is exactly d at every site
      • Bond dimensions are positive integers and do not exceed chi
      • Neighboring tensors have matching internal bond sizes:
            right_bond(As[i-1]) == left_bond(As[i])
      • Shapes:
            First:    (d, r1)                         with 1 <= r1 <= chi
            Interior: (r_i, d, r_{i+1})              with 1 <= r_i, r_{i+1} <= chi
            Last:
                - if out_dim == 1: (r_{L-1}, d) or (r_{L-1}, d, 1)
                - else:            (r_{L-1}, d, out_dim)
    """
    if len(As) != L:
        raise ValueError(f"As must have length L={L}, got {len(As)}.")
    if L < 2:
        raise ValueError("MPS requires L >= 2 (first, interior..., last).")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")
    if chi <= 0:
        raise ValueError(f"chi must be positive, got {chi}.")
    if out_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}.")

    # ---------- First ----------
    s0 = tuple(As[0].shape)
    if len(s0) != 2:
        raise ValueError(f"As[0] must be rank-2 with shape (d, r), got {s0}.")
    if s0[0] != d:
        raise ValueError(f"As[0] physical dimension must be d={d} on axis 0, got {s0[0]}.")
    r_prev = s0[1]
    if r_prev < 1 or r_prev > chi:
        raise ValueError(
            f"As[0] bond dimension r1 must satisfy 1 <= r1 <= chi={chi}, got r1={r_prev}."
        )

    # ---------- Interiors ----------
    for i in range(1, L - 1):
        si = tuple(As[i].shape)
        if len(si) != 3:
            raise ValueError(f"As[{i}] must be rank-3 with shape (l, d, r), got {si}.")
        l, di, r = si
        if di != d:
            raise ValueError(
                f"As[{i}] physical dimension must be d={d} on axis 1, got {di}."
            )
        if l != r_prev:
            raise ValueError(
                f"Bond mismatch between As[{i-1}] and As[{i}]: "
                f"right_bond(As[{i-1}])={r_prev} != left_bond(As[{i}])={l}."
            )
        if l < 1 or r < 1 or l > chi or r > chi:
            raise ValueError(
                f"As[{i}] bond dims must satisfy 1 <= l,r <= chi={chi}, got (l,r)=({l},{r})."
            )
        r_prev = r  # right bond to match next left bond

    # ---------- Last ----------
    sL = tuple(As[L - 1].shape)
    if out_dim == 1:
        if len(sL) == 2:
            l, d_last = sL
            if d_last != d:
                raise ValueError(
                    f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
                )
            if l != r_prev:
                raise ValueError(
                    f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                    f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
                )
            if l < 1 or l > chi:
                raise ValueError(
                    f"As[L-1] left bond must satisfy 1 <= l <= chi={chi}, got l={l}."
                )
        elif len(sL) == 3:
            l, d_last, o = sL
            if d_last != d:
                raise ValueError(
                    f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
                )
            if o != 1:
                raise ValueError(
                    f"As[L-1] output dimension must be 1 when out_dim==1, got {o}."
                )
            if l != r_prev:
                raise ValueError(
                    f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                    f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
                )
            if l < 1 or l > chi:
                raise ValueError(
                    f"As[L-1] left bond must satisfy 1 <= l <= chi={chi}, got l={l}."
                )
        else:
            raise ValueError(
                f"As[L-1] must have shape (l, d) or (l, d, 1) when out_dim==1, got {sL}."
            )
    else:
        if len(sL) != 3:
            raise ValueError(
                f"As[L-1] must be rank-3 with shape (l, d, {out_dim}) when out_dim={out_dim}, got {sL}."
            )
        l, d_last, o = sL
        if d_last != d:
            raise ValueError(
                f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
            )
        if o != out_dim:
            raise ValueError(
                f"As[L-1] output dimension must be out_dim={out_dim}, got {o}."
            )
        if l != r_prev:
            raise ValueError(
                f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
            )
        if l < 1 or l > chi:
            raise ValueError(
                f"As[L-1] left bond must satisfy 1 <= l <= chi={chi}, got l={l}."
            )


def _construct_core_As(
    L: int,
    d: Union[int, Sequence[int]],
    chi: Union[int, Sequence[int]],
    out_dim: int = 1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> List[torch.Tensor]:
    """
    Construct MPS cores with (optionally) site-dependent local and bond dimensions.

    Open boundary layout:
        As[0]      : (d_0, chi_0)
        As[i]      : (chi_{i-1}, d_i, chi_i)   for i = 1 .. L-2
        As[L-1]    : (chi_{L-2}, d_{L-1})      if out_dim == 1
                   : (chi_{L-2}, d_{L-1}, out_dim) otherwise

    If `d` is an int, use d_i ≡ d for all sites.
    If `chi` is an int, use chi_i ≡ chi for all bonds.
    """

    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")

    # Normalize d into a list of length L
    if isinstance(d, int):
        d_list = [d] * L
    else:
        d_list = list(d)
        if len(d_list) != L:
            raise ValueError(f"`d` must have length L={L}, got {len(d_list)}")

    # Normalize chi into a list of length L-1 (bonds)
    if L == 1:
        chi_list: List[int] = []
    else:
        if isinstance(chi, int):
            chi_list = [chi] * (L - 1)
        else:
            chi_list = list(chi)
            if len(chi_list) != L - 1:
                raise ValueError(
                    f"`chi` must have length L-1={L-1}, got {len(chi_list)}"
                )

    As: List[torch.Tensor] = []

    # Single-site special case
    if L == 1:
        if out_dim == 1:
            As.append(torch.zeros(d_list[0], device=device, dtype=dtype))
        else:
            As.append(torch.zeros(d_list[0], out_dim, device=device, dtype=dtype))
        return As

    # First core: (d_0, chi_0)
    As.append(torch.zeros(d_list[0], chi_list[0], device=device, dtype=dtype))

    # Interior cores: (chi_{i-1}, d_i, chi_i)
    for i in range(1, L - 1):
        As.append(
            torch.zeros(
                chi_list[i - 1], d_list[i], chi_list[i],
                device=device, dtype=dtype
            )
        )

    # Last core
    if out_dim == 1:
        As.append(
            torch.zeros(
                chi_list[L - 2], d_list[L - 1],
                device=device, dtype=dtype
            )
        )
    else:
        As.append(
            torch.zeros(
                chi_list[L - 2], d_list[L - 1], out_dim,
                device=device, dtype=dtype
            )
        )

    return As

def _infer_bond_dims_from_As(
    L: int,
    d: int,
    out_dim: int,
    As: Sequence[torch.Tensor],
) -> List[int]:
    """
    Infer per-bond dimensions [chi_0, ..., chi_{L-2}] from a list of cores As.

    Uses the same shape conventions as `_validate_shapes_base` but does not
    require a chi bound. Raises if shapes are inconsistent with (L, d, out_dim).
    """
    if len(As) != L:
        raise ValueError(f"As must have length L={L}, got {len(As)}.")
    if L < 2:
        raise ValueError("MPS requires L >= 2 (first, interior..., last).")

    # ---- First core ----
    s0 = tuple(As[0].shape)
    if len(s0) != 2:
        raise ValueError(f"As[0] must be rank-2 with shape (d, r1), got {s0}.")
    if s0[0] != d:
        raise ValueError(f"As[0] physical dimension must be d={d} on axis 0, got {s0[0]}.")
    r_prev = s0[1]
    bonds: List[int] = [r_prev]

    # ---- Interior cores ----
    for i in range(1, L - 1):
        si = tuple(As[i].shape)
        if len(si) != 3:
            raise ValueError(f"As[{i}] must be rank-3 with shape (l, d, r), got {si}.")
        l, di, r = si
        if di != d:
            raise ValueError(f"As[{i}] physical dimension must be d={d} on axis 1, got {di}.")
        if l != r_prev:
            raise ValueError(
                f"Bond mismatch between As[{i-1}] and As[{i}]: "
                f"right_bond(As[{i-1}])={r_prev} != left_bond(As[{i}])={l}."
            )
        bonds.append(r)
        r_prev = r

    # ---- Last core ----
    sL = tuple(As[L - 1].shape)
    if out_dim == 1:
        if len(sL) == 2:
            l, d_last = sL
            if d_last != d:
                raise ValueError(
                    f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
                )
            if l != r_prev:
                raise ValueError(
                    f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                    f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
                )
        elif len(sL) == 3:
            l, d_last, o = sL
            if d_last != d:
                raise ValueError(
                    f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
                )
            if o != 1:
                raise ValueError(
                    f"As[L-1] output dimension must be 1 when out_dim==1, got {o}."
                )
            if l != r_prev:
                raise ValueError(
                    f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                    f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
                )
        else:
            raise ValueError(
                f"As[L-1] must have shape (l, d) or (l, d, 1) when out_dim==1, got {sL}."
            )
    else:
        if len(sL) != 3:
            raise ValueError(
                f"As[L-1] must be rank-3 with shape (l, d, {out_dim}) when out_dim={out_dim}, got {sL}."
            )
        l, d_last, o = sL
        if d_last != d:
            raise ValueError(
                f"As[L-1] physical dimension must be d={d} on axis 1, got {d_last}."
            )
        if o != out_dim:
            raise ValueError(
                f"As[L-1] output dimension must be out_dim={out_dim}, got {o}."
            )
        if l != r_prev:
            raise ValueError(
                f"Bond mismatch between As[{L-2}] and As[{L-1}]: "
                f"right_bond(As[{L-2}])={r_prev} != left_bond(As[{L-1}])={l}."
            )

    return bonds

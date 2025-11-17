
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import math
import torch
import os
from typing import Dict, Any
import opt_einsum as oe  # type: ignore[import-untyped]

from ..opt_einsum_utils import GetSymbolFn, _EqAndPathCache
from ..mpsqsc.mpstate import MPState


class qMPS:
    """
    qMPS: Embed an MPS classifier into a quantum circuit made of 4‑leg unitaries.

    Layout
    ------
    For L sites, we expect L-1 4‑leg blocks U[i] with shapes:
      U[0]      : (a0, chi0, d, d)
      U[i]      : (ai, chii, chi_{i-1}, d)        for 1 <= i <= L-2
      U[L-2]    : (a_{L-1}, C, chi_{L-2}, d)      (C = out_dim)

    We also have a "last_unitary" of shape (2, C, 2, C) that acts on
    (ancilla, class), with ancilla post-selected to |0⟩ on its input leg.

    Contraction computed by `_contract_circuit_with_state(state)` returns a
    (C, C) tensor (class Gram / score matrix) after partially tracing out
    all internal degrees of freedom and post-selecting ancilla=0 on the
    input leg of the last unitary on both rails.
    """

    def __init__(
        self,
        L: int,
        chi: int,
        d: int,
        Us: List[torch.Tensor],
        last_unitary: Optional[torch.Tensor] = None,
        out_dim: int = 2,
        As: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device | str] = None,
        seed: Optional[int] = None,
        optimize: str = "random-greedy",
    ):
        if L < 2:
            raise ValueError("MPS requires L >= 2.")
        if out_dim < 2:
            raise ValueError("out_dim must be >= 2.")

        self.L = L
        self.chi = chi
        self.d = d
        self.out_dim = out_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = Us[0].dtype
        self.optimize = optimize

        # --- FIX: default last_unitary must be (2*out_dim)×(2*out_dim) so it reshapes to (2, C, 2, C)
        if last_unitary is None:
            last_unitary = torch.eye(2 * out_dim, dtype=Us[0].dtype, device=Us[0].device)
        self._check_unitary(last_unitary, atol=1e-7, rtol=1e-7, i=L)  # label i=L for clarity
        self.last_unitary = last_unitary.reshape(2, out_dim, 2, out_dim)

        # Caches for classifier contraction (equation + path)
        self._cls_cache = _EqAndPathCache()

        # Store and validate the per-site unitaries; convert to 4‑leg tensors
        self.Us = [U.clone().detach().to(self.device) for U in Us]
        self.U4, self.ancillas, self.chis = self.validate_and_reshape_Us(Us=self.Us)
        for U in self.U4:
            U.requires_grad_(True)
        self.last_unitary = self.last_unitary.requires_grad_(True)
        self.weights = [torch.zeros(dim_ancilla, dtype=Us[0].dtype, device=Us[0].device) for dim_ancilla in self.ancillas]
        self.weights.append(torch.zeros(2, dtype=Us[0].dtype, device=Us[0].device))
        for w in self.weights:
            w[0] = 1.0
        for w in self.weights:
            w.requires_grad_(True)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _check_unitary(self, U: torch.Tensor, *, atol: float, rtol: float, i: int) -> None:
        """
        Validate unitarity of a square matrix U within (atol, rtol).

        Raises a ValueError with a readable diagnostic on failure.
        """
        if U.dtype != self.dtype:
            raise ValueError(f"U[{i}] must be of dtype {self.dtype}, got {U.dtype}.")
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError(f"U[{i}] must be a square 2D matrix, got {tuple(U.shape)}.")
        n = U.shape[0]
        eye = torch.eye(n, dtype=U.dtype, device=U.device)
        Uh = U.mH
        if not torch.allclose(U @ Uh, eye, atol=atol, rtol=rtol):
            err = torch.linalg.norm(U @ Uh - eye, ord="fro").item()
            raise ValueError(f"U[{i}] fails U U^† ≈ I: ||U U^† - I||_F = {err:.3e}")
        if not torch.allclose(Uh @ U, eye, atol=atol, rtol=rtol):
            err = torch.linalg.norm(Uh @ U - eye, ord="fro").item()
            raise ValueError(f"U[{i}] fails U^† U ≈ I: ||U^† U - I||_F = {err:.3e}")

    # ---------------------------------------------------------------------
    # Naming rule + circuit equation
    # ---------------------------------------------------------------------
    def _build_equation_circuit(self) -> tuple[str, dict[str, List[str] | str], GetSymbolFn]:
        """
        Construct the einsum equation for the *circuit rails* only (no state yet).

        Index naming rule
        -----------------
        phys_L[i], phys_R[i] : physical legs for the left/right rail (i = 0..L-1)
        bond_L[i], bond_R[i] : internal χ bonds on left/right rail (i = 0..L-2)
        anc[i]               : ancilla leg at layer i (i = 0..L-1)
        cls_L, cls_R         : class legs produced at the final layer (one per rail)

        Pieces
        ------
        Left rail:
          U[0]      : anc[0] bond_L[0] phys_L[0] phys_L[1]
          U[i]      : anc[i] bond_L[i] bond_L[i-1] phys_L[i+1]      (1 <= i <= L-2)
          last op   : anc[L-1] cls_L bond_L[L-2]                    (*see note)

        Right rail: identical with bond_R/phys_R and cls_R.

        Note: As in your current design, the "last unitary" is used
        with its input ancilla fixed to |0⟩; we handle that by slicing
        before contraction. Its remaining indices are matched here as
        (anc_out, cls_out, ???), consistent with your prior code.
        """
        sym = GetSymbolFn()
        L = self.L

        syms: Dict[str, List[str] | str] = {}
        syms["phys_L"] = [sym() for _ in range(L)]
        syms["phys_R"] = [sym() for _ in range(L)]
        syms["anc"]    = [sym() for _ in range(L)]
        syms["bond_L"] = [sym() for _ in range(L - 1)]
        syms["bond_R"] = [sym() for _ in range(L - 1)]
        syms["cls_L"]  = sym()
        syms["cls_R"]  = sym()

        left_pieces: List[str] = []
        right_pieces: List[str] = []

        # Left rail unitaries (U4)
        left_pieces.append(syms["anc"][0] + syms["bond_L"][0] + syms["phys_L"][0] + syms["phys_L"][1])
        for i in range(1, L - 1):
            left_pieces.append(
                syms["anc"][i] + syms["bond_L"][i] + syms["bond_L"][i - 1] + syms["phys_L"][i + 1]
            )
        # Last op on left rail (matches your slicing usage)
        left_pieces.append(syms["anc"][L - 1] + syms["cls_L"] + syms["bond_L"][L - 2]) # type: ignore

        # Right rail unitaries (conjugate rail)
        right_pieces.append(syms["anc"][0] + syms["bond_R"][0] + syms["phys_R"][0] + syms["phys_R"][1])
        for i in range(1, L - 1):
            right_pieces.append(
                syms["anc"][i] + syms["bond_R"][i] + syms["bond_R"][i - 1] + syms["phys_R"][i + 1]
            )
        # Last op on right rail
        right_pieces.append(syms["anc"][L - 1] + syms["cls_R"] + syms["bond_R"][L - 2]) # type: ignore

        eq = ",".join(left_pieces) + "," + ",".join(right_pieces) + "->" + syms["cls_L"] + syms["cls_R"] # type: ignore
        return eq, syms, sym

    # ---------------------------------------------------------------------
    # Validation + 4‑leg reshaping (kept as you wrote; docstring clarified)
    # ---------------------------------------------------------------------
    def validate_and_reshape_Us(
        self,
        Us: Optional[List[torch.Tensor]] = None,
        *,
        atol: float = 1e-7,
        rtol: float = 1e-5,
    ):
        """
        Validate chain and reshape each U_i (square) into a 4‑leg tensor (out_left, out_right, in_left, in_right).

        Returns:
          U4        : list of 4‑leg tensors in the order required by `_build_equation_circuit`.
          ancillas  : [a0, a1, ..., a_{L-1}]      (ancilla/output-left sizes)
          chis      : [chi0, ..., chi_{L-2}, C]   (bond/output-right sizes, last is C)
        """
        if Us is None:
            Us = self.Us
        if len(Us) != self.L - 1:
            raise ValueError(f"Expected L-1={self.L-1} unitaries, got {len(Us)}.")

        L, d, C, chi_cap = self.L, self.d, self.out_dim, self.chi

        # --- Unitarity checks and collect sizes ---
        sizes = []
        for i, U in enumerate(Us):
            self._check_unitary(U, atol=atol, rtol=rtol, i=i)
            sizes.append(int(U.shape[0]))  # = U.shape[1]

        dims : List[Dict[str, Tuple[int, int]]] = []  # {"in": (inL, inR), "out": (outL, outR)}
        chis : List[int] = []
        ancillas : List[int] = []

        # Site 0
        N0 = sizes[0]
        expected0 = d * d
        if N0 != expected0:
            raise ValueError(f"U[0] size must be d*d={expected0}, got {N0}.")
        if sizes[1] % d != 0:
            raise ValueError(f"U[1] size must be divisible by d={d} to define chi_0.")
        chi0 = sizes[1] // d
        if not (chi0 <= chi_cap):
            raise ValueError(f"chi_0={chi0} must be <= chi={chi_cap}.")
        if N0 % chi0 != 0:
            raise ValueError(f"d*d={N0} must be divisible by chi_0={chi0}.")
        a0 = N0 // chi0
        dims.append({"in": (d, d), "out": (a0, chi0)})
        chis.append(chi0)
        ancillas.append(a0)
        chi_prev = chi0

        # Sites 1..L-2
        for i in range(1, L - 2):
            Ni = sizes[i]
            expected_in = chi_prev * d
            if Ni != expected_in:
                raise ValueError(
                    f"U[{i}] size must be chi_{i-1}*d={chi_prev}*{d}={expected_in}, got {Ni}."
                )
            if sizes[i + 1] % d != 0:
                raise ValueError(f"U[{i+1}] size must be divisible by d={d} to define chi_{i}.")
            chii = sizes[i + 1] // d
            if not (chii <= chi_cap):
                raise ValueError(f"chi_{i}={chii} must be <= chi={chi_cap}.")
            if Ni % chii != 0:
                raise ValueError(f"U[{i}] size {Ni} must be divisible by chi_{i}={chii}.")
            ai = Ni // chii
            dims.append({"in": (chi_prev, d), "out": (ai, chii)})
            chis.append(chii)
            ancillas.append(ai)
            chi_prev = chii

        # Last U (index L-2 in Us)
        Nlast = sizes[L - 2]
        expected_last_in = chi_prev * d
        if Nlast != expected_last_in:
            raise ValueError(
                f"U[{L-2}] size must be chi_{L-2}*d={chi_prev}*{d}={expected_last_in}, got {Nlast}."
            )
        if Nlast % C != 0:
            raise ValueError(f"U[{L-2}] size {Nlast} must be divisible by out_dim={C}.")
        a_last = Nlast // C
        dims.append({"in": (chi_prev, d), "out": (a_last, C)})
        ancillas.append(a_last)
        chis.append(C)

        # Reshape into (outL, outR, inL, inR)
        U4 = []
        for i, U in enumerate(Us):
            outL, outR = dims[i]["out"]
            inL, inR   = dims[i]["in"]
            U4.append(U.reshape(outL, outR, inL, inR))

        return U4, ancillas, chis

    # ---------------------------------------------------------------------
    # Cached equation+path for partial trace contraction
    # ---------------------------------------------------------------------
    def _build_partial_trace_path(self, state: MPState) -> tuple[str, oe.Path]:
        """
        Build (and cache) the full einsum equation and an optimized path for contracting:
            ⟨ψ|  (Circuit† with ancilla post-selection) (Circuit with ancilla post-selection) |ψ⟩
        resulting in a (C, C) tensor over the two class legs.

        Caching key
        -----------
        The cache key is the tuple of shapes of all tensors in the contraction:
        [state.As, state.As_conj, U4, last_op, U4_conj, last_op_conj].

        Returns
        -------
        eq : str
            The combined einsum equation across states and both rails.
        path : oe.Path
            The optimized contraction path from `opt_einsum.contract_path`.
        """
        # --- Circuit-only equation with new naming rule
        eq_circ, syms, sym_circ = self._build_equation_circuit()

        # --- Build state's equations (two copies) and shift symbols to avoid collisions
        eq_stateL, syms_stateL, sym_stateL = state._build_full_equation()   # "...->phys_indices"
        eq_stateR, syms_stateR, sym_stateR = state._build_full_equation()

        eq_stateL = eq_stateL.split("->")[0]
        eq_stateR = eq_stateR.split("->")[0]

        # Shift state symbols to be disjoint from circuit + prior state
        eq_stateL, syms_stateL = sym_stateL.shift_chars(sym_circ.symbol_count, eq_stateL, syms_stateL)
        eq_stateR, syms_stateR = sym_stateR.shift_chars(sym_circ.symbol_count + sym_stateL.symbol_count,
                                                        eq_stateR, syms_stateR)

        # Map state physical symbols to our circuit naming (phys_L / phys_R)
        for p_sym, phys_sym in zip(syms_stateL["p"], syms["phys_L"]):  # type: ignore[index]
            eq_stateL = eq_stateL.replace(p_sym, phys_sym)
        for p_sym, phys_sym in zip(syms_stateR["p"], syms["phys_R"]):  # type: ignore[index]
            eq_stateR = eq_stateR.replace(p_sym, phys_sym)

        # Combine: stateL, stateR, circuit
        eq_full = eq_stateL + "," + eq_stateR + "," + eq_circ

        # Tensors in the same order as eq_full:
        #   [state.As] + [state.As†] + [U4(left rail)] + [last_op] + [U4(right rail)†] + [last_op†]
        last_operator = self.last_unitary[:, :, 0, :]  # (2, C, C) after post-select ancilla input = |0⟩

        tensors = (
            [A for A in state.As] +
            [A.conj() for A in state.As] +
            [U for U in self.U4] + [last_operator] +
            [U.conj() for U in self.U4] + [last_operator.conj()]
        )

        shape_sig = tuple(tuple(T.shape) for T in tensors)
        if self._cls_cache.shape_sig != shape_sig:
            path, _ = oe.contract_path(eq_full, *tensors, optimize=self.optimize)
            self._cls_cache.eq = eq_full
            self._cls_cache.path = path
            self._cls_cache.shape_sig = shape_sig

        # Sanity
        if self._cls_cache.eq is None or self._cls_cache.path is None:
            raise RuntimeError("Classifier equation/path was not built.")
        return self._cls_cache.eq, self._cls_cache.path

    # ---------------------------------------------------------------------
    # Public contraction
    # ---------------------------------------------------------------------
    def _contract_circuit_with_state_partial_trace(self, state: MPState) -> torch.Tensor:
        """
        Contract two copies of the input state with the qMPS circuit to produce a (C, C) tensor.

        This should return the density matrix.

        The contraction does:
          1) Two rails (left/right) of the circuit built from U4 blocks,
          2) Two copies of the state (bra/ket),
          3) Post-selection of ancilla input = |0⟩ at the last gate of each rail,
          4) Partial trace over all internal indices except the two class legs.

        Returns
        -------
        torch.Tensor
            A tensor of shape (out_dim, out_dim), i.e., (C, C).
        """
        eq, path = self._build_partial_trace_path(state)

        last_operator = self.last_unitary[:, :, 0, :]
        tensors = (
            [A for A in state.As] +
            [A.conj() for A in state.As] +
            [U for U in self.U4] + [last_operator] +
            [U.conj() for U in self.U4] + [last_operator.conj()]
        )
        return oe.contract(eq, *tensors, optimize=path)

    # ---------------------------------------------------------------------
    # Cached equation+path for adiabatic encoding contraction
    # ---------------------------------------------------------------------
    def _build_ae_path(self, state: MPState) -> tuple[str, oe.Path]:
        """
        Build (and cache) the full einsum equation and an optimized path for contracting:
            ⟨ψ|  (Circuit† with ancilla post-selection) (Circuit with ancilla post-selection) |ψ⟩
        resulting in a (C, C) tensor over the two class legs.

        Caching key
        -----------
        The cache key is the tuple of shapes of all tensors in the contraction:
        [state.As, state.As_conj, U4, last_op, U4_conj, last_op_conj].

        Returns
        -------
        eq : str
            The combined einsum equation across states and both rails.
        path : oe.Path
            The optimized contraction path from `opt_einsum.contract_path`.
        """
        # --- Circuit-only equation with new naming rule
        eq_circ, syms, sym_circ = self._build_equation_ae()

        # --- Build state's equations (two copies) and shift symbols to avoid collisions
        eq_stateL, syms_stateL, sym_stateL = state._build_full_equation()   # "...->phys_indices"
        eq_stateR, syms_stateR, sym_stateR = state._build_full_equation()

        eq_stateL = eq_stateL.split("->")[0]
        eq_stateR = eq_stateR.split("->")[0]

        # Shift state symbols to be disjoint from circuit + prior state
        eq_stateL, syms_stateL = sym_stateL.shift_chars(sym_circ.symbol_count, eq_stateL, syms_stateL)
        eq_stateR, syms_stateR = sym_stateR.shift_chars(sym_circ.symbol_count + sym_stateL.symbol_count,
                                                        eq_stateR, syms_stateR)

        # Map state physical symbols to our circuit naming (phys_L / phys_R)
        for p_sym, phys_sym in zip(syms_stateL["p"], syms["phys_L"]):  # type: ignore[index]
            eq_stateL = eq_stateL.replace(p_sym, phys_sym)
        for p_sym, phys_sym in zip(syms_stateR["p"], syms["phys_R"]):  # type: ignore[index]
            eq_stateR = eq_stateR.replace(p_sym, phys_sym)

        # Combine: stateL, stateR, circuit
        eq_full = eq_stateL + "," + eq_stateR + "," + eq_circ

        # Tensors in the same order as eq_full:
        #   [state.As] + [state.As†] + [U4(left rail)] + [last_op] + [U4(right rail)†] + [last_op†]
        last_operator = self.last_unitary[:, :, 0, :]  # (2, C, C) after post-select ancilla input = |0⟩

        tensors = (
            [A for A in state.As] +
            [A.conj() for A in state.As] +
            [U for U in self.U4] + [last_operator] +
            [U.conj() for U in self.U4] + [last_operator.conj()] +
            [w for w in self.weights]
        )

        shape_sig = tuple(tuple(T.shape) for T in tensors)
        if self._cls_cache.shape_sig != shape_sig:
            path, _ = oe.contract_path(eq_full, *tensors, optimize=self.optimize)
            self._cls_cache.eq = eq_full
            self._cls_cache.path = path
            self._cls_cache.shape_sig = shape_sig

        # Sanity
        if self._cls_cache.eq is None or self._cls_cache.path is None:
            raise RuntimeError("Classifier equation/path was not built.")
        return self._cls_cache.eq, self._cls_cache.path

    def _build_equation_ae(self) -> tuple[str, dict[str, List[str] | str], GetSymbolFn]:
        """
        Construct the einsum equation for the *circuit + adiabatic encoding rails* only (no state yet).

        Index naming rule
        -----------------
        phys_L[i], phys_R[i] : physical legs for the left/right rail (i = 0..L-1)
        bond_L[i], bond_R[i] : internal χ bonds on left/right rail (i = 0..L-2)
        anc[i]               : ancilla leg at layer i (i = 0..L-1)
        cls_L, cls_R         : class legs produced at the final layer (one per rail)

        Pieces
        ------
        Left rail:
          U[0]      : anc[0] bond_L[0] phys_L[0] phys_L[1]
          U[i]      : anc[i] bond_L[i] bond_L[i-1] phys_L[i+1]      (1 <= i <= L-2)
          last op   : anc[L-1] cls_L bond_L[L-2]                    (*see note)

        Right rail: identical with bond_R/phys_R and cls_R.

        Note: As in your current design, the "last unitary" is used
        with its input ancilla fixed to |0⟩; we handle that by slicing
        before contraction. Its remaining indices are matched here as
        (anc_out, cls_out, ???), consistent with your prior code.
        """
        sym = GetSymbolFn()
        L = self.L

        syms: Dict[str, List[str] | str] = {}
        syms["phys_L"] = [sym() for _ in range(L)]
        syms["phys_R"] = [sym() for _ in range(L)]
        syms["anc"]    = [sym() for _ in range(L)]
        syms["bond_L"] = [sym() for _ in range(L - 1)]
        syms["bond_R"] = [sym() for _ in range(L - 1)]
        syms["cls_L"]  = sym()
        syms["cls_R"]  = sym()

        left_pieces: List[str] = []
        right_pieces: List[str] = []

        # Left rail unitaries (U4)
        left_pieces.append(syms["anc"][0] + syms["bond_L"][0] + syms["phys_L"][0] + syms["phys_L"][1])
        for i in range(1, L - 1):
            left_pieces.append(
                syms["anc"][i] + syms["bond_L"][i] + syms["bond_L"][i - 1] + syms["phys_L"][i + 1]
            )
        # Last op on left rail (matches your slicing usage)
        left_pieces.append(syms["anc"][L - 1] + syms["cls_L"] + syms["bond_L"][L - 2]) # type: ignore

        # Right rail unitaries (conjugate rail)
        right_pieces.append(syms["anc"][0] + syms["bond_R"][0] + syms["phys_R"][0] + syms["phys_R"][1])
        for i in range(1, L - 1):
            right_pieces.append(
                syms["anc"][i] + syms["bond_R"][i] + syms["bond_R"][i - 1] + syms["phys_R"][i + 1]
            )
        # Last op on right rail
        right_pieces.append(syms["anc"][L - 1] + syms["cls_R"] + syms["bond_R"][L - 2]) # type: ignore


        eq = (
            ",".join(left_pieces)  # type: ignore
            + ","
            + ",".join(right_pieces)
            + ","
            + ",".join(syms["anc"])
            + "->"
            + syms["cls_L"]
            + syms["cls_R"]
        )  # type: ignore
        return eq, syms, sym


    def _contract_circuit_with_state_ae(self, state: MPState) -> torch.Tensor:
        """
        Contract two copies of the input state with the qMPS circuit to produce a (C, C) tensor.

        This should return the density matrix.

        The contraction does:
          1) Two rails (left/right) of the circuit built from U4 blocks,
          2) Two copies of the state (bra/ket),
          3) Post-selection of ancilla input = |0⟩ at the last gate of each rail,
          4) Partial trace over all internal indices except the two class legs.

        Returns
        -------
        torch.Tensor
            A tensor of shape (out_dim, out_dim), i.e., (C, C).
        """
        eq, path = self._build_ae_path(state)

        last_operator = self.last_unitary[:, :, 0, :]
        tensors = (
            [A for A in state.As] +
            [A.conj() for A in state.As] +
            [U for U in self.U4] + [last_operator] +
            [U.conj() for U in self.U4] + [last_operator.conj()] +
            [w for w in self.weights]
        )
        return oe.contract(eq, *tensors, optimize=path)
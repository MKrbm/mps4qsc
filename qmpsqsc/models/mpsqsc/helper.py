import torch
from typing import List, Sequence
from .mpstate import MPState
from .mpsqsc import MpsQsc
from .mpsbase import _construct_core_As


def build_ghz_state(L: int, d: int, chi: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float64) -> MPState:
    """
    Build a GHZ state for an MPState.
    """

    As: List[torch.Tensor] = []

    # First core: A0(s, a1)
    # s=0 -> bond=0; s=1 -> bond=1
    A0 = torch.zeros(d, chi, device=device, dtype=dtype)
    A0[0, 0] = 1.0
    A0[1, 1] = 1.0
    As.append(A0)

    # Interior cores: A_mid(a_{i-1}, s, a_i)
    # branch 0 stays in 0 when s=0; branch 1 stays in 1 when s=1.
    for _ in range(1, L - 1):
        A_mid = torch.zeros(chi, d, chi, device=device, dtype=dtype)
        A_mid[0, 0, 0] = 1.0  # |0> keeps branch 0
        A_mid[1, 1, 1] = 1.0  # |1> keeps branch 1
        As.append(A_mid)

    # Last core: AL(a_{L-1}, s_L)
    # Only valid if s_L matches branch.
    AL = torch.zeros(chi, d, device=device, dtype=dtype)
    AL[0, 0] = 1.0
    AL[1, 1] = 1.0
    As.append(AL)

    mpstate = MPState(L=L, chi=chi, d=d, As=As, device=device, dtype=dtype)

    return mpstate

def build_product_state(
    L: int,
    d: int,
    state: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> MPState:

    """
    Build a product state for an MPState.
    """

    state = state.to(device, dtype)
    # check if state is a tensor of shape (d,)*L
    if state.shape != (L, d):
        raise ValueError(f"State must have shape (d,)*L={d}**L, got {state.shape}")

    As = _construct_core_As(L, d, 1, out_dim=1, device=device, dtype=dtype)

    As[0][:, 0] = state[0]
    for i in range(1, L-1):
        As[i][0, :, 0] = state[i]
    As[L-1][0, :] = state[L-1]
    mpstate = MPState(L=L, chi=1, d=d, As=As, device=device, dtype=dtype)
    return mpstate

def build_classical_state(
    L: int,
    d: int,
    bits: List[int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> MPState:

    """
    Build a product state for an MPState.
    """

    # create a state tensor of shape (L, d)
    state = torch.zeros(L, d, device=device, dtype=dtype)
    for i in range(L):
        state[i, bits[i]] = 1.0

    return build_product_state(L, d, state, device, dtype)



# Helper functions for mpsqsc
def build_qsc_from_mpstate(mpstate: MPState) -> MpsQsc:
    """
    Build a MpsQsc from an MPState.
    """

    As = mpstate.As
    chi = mpstate.chi_max
    d = mpstate.d
    L = mpstate.L

    As_qsc = [A.clone() for A in As]
    As_qsc[-1] = torch.zeros(chi, d, 2, device=mpstate.device, dtype=mpstate.dtype)
    As_qsc[-1][:, :, 0] = As[-1]

    mpsqsc = MpsQsc(L, chi, d, As_qsc, device=mpstate.device, dtype=mpstate.dtype)
    return mpsqsc


def build_2qsc_from_mpstate(mpstate1: MPState, mpstate2: MPState) -> MpsQsc:
    """
    Build a MpsQsc from an MPState.
    """

    As1 = mpstate1.As
    chi1 = mpstate1.chi_max
    d1 = mpstate1.d
    L1 = mpstate1.L

    As2 = mpstate2.As
    chi2 = mpstate2.chi_max
    d2 = mpstate2.d
    L2 = mpstate2.L

    assert L1 == L2
    assert d1 == d2

    chi = chi1 + chi2
    As = _construct_core_As(L1, d1, chi, out_dim=2, device=mpstate1.device, dtype=mpstate1.dtype)
    As[0][:, :chi1] = As1[0]
    As[0][:, chi1:] = As2[0]
    for i in range(1, L1 - 1):
        As[i][:chi1, :, :chi1] = As1[i]
        As[i][chi1:, :, chi1:] = As2[i]
    As[L1 - 1][:chi1, :, 0] = As1[L1 - 1]
    As[L1 - 1][chi1:, :, 1] = As2[L1 - 1]

    mpsqsc = MpsQsc(L1, chi, d1, As, device=mpstate1.device, dtype=mpstate1.dtype)
    return mpsqsc

def add_mpstates(mpstates: List[MPState]) -> MPState:
    """
    Direct-sum (superposition) of multiple MPStates, allowing for
    site-dependent bond dimensions.

    Requirements:
        * All states have the same device, dtype, L and d.
        * Each state's cores have shapes:
              As[0]      : (d, chi_0)
              As[i]      : (chi_{i-1}, d, chi_i)  for i = 1 .. L-2
              As[L-1]    : (chi_{L-2}, d)
    """

    if not mpstates:
        raise ValueError("mpstates must be a non-empty list")

    ref = mpstates[0]
    dtype = ref.dtype
    device = ref.device
    L = ref.L
    d = ref.d

    # Consistency checks
    for mp in mpstates[1:]:
        if mp.dtype != dtype:
            raise ValueError(
                f"All mpstates must have the same dtype, got {[m.dtype for m in mpstates]}"
            )
        if mp.device != device:
            raise ValueError(
                f"All mpstates must have the same device, got {[m.device for m in mpstates]}"
            )
        if mp.L != L:
            raise ValueError(
                f"All mpstates must have the same L, got {[m.L for m in mpstates]}"
            )
        if mp.d != d:
            raise ValueError(
                f"All mpstates must have the same d, got {[m.d for m in mpstates]}"
            )

    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")

    # Single-site case: just sum the local vectors
    if L == 1:
        A0 = mpstates[0].As[0].clone()
        for mp in mpstates[1:]:
            A0 = A0 + mp.As[0]

        return MPState(
            L=1,
            chi=1,  # effectively no internal bond
            d=d,
            As=[A0],
            device=device,
            dtype=dtype,
        )

    num_states = len(mpstates)
    num_bonds = L - 1

    # Extract bond dimensions for each state
    bond_dims_per_state: List[List[int]] = []
    for mp in mpstates:
        As = mp.As
        if len(As) != L:
            raise ValueError(
                f"mpstate.As length {len(As)} does not match L={L}"
            )

        bonds: List[int] = []

        # Bond between site 0 and 1
        A0 = As[0]
        if A0.ndim != 2:
            raise ValueError(
                f"Expected A[0] to have shape (d, chi_0), got {tuple(A0.shape)}"
            )
        bonds.append(A0.shape[1])

        # Bonds between site i and i+1, for i = 1 .. L-2
        for i in range(1, L - 1):
            Ai = As[i]
            if Ai.ndim != 3:
                raise ValueError(
                    f"Expected A[{i}] to have shape (chi_{i-1}, d, chi_i), "
                    f"got {tuple(Ai.shape)}"
                )
            left = Ai.shape[0]
            right = Ai.shape[2]

            # Check that left dim matches previous bond
            if left != bonds[i - 1]:
                raise ValueError(
                    f"Inconsistent left bond dimension at site {i}: "
                    f"got {left}, expected {bonds[i - 1]}"
                )

            bonds.append(right)

        # Last core: (chi_{L-2}, d)
        A_last = As[L - 1]
        if A_last.ndim != 2:
            raise ValueError(
                f"Expected A[L-1] to have shape (chi_{L-2}, d), "
                f"got {tuple(A_last.shape)}"
            )
        if A_last.shape[0] != bonds[-1]:
            raise ValueError(
                f"Inconsistent left bond dimension at last site: "
                f"got {A_last.shape[0]}, expected {bonds[-1]}"
            )

        bond_dims_per_state.append(bonds)

    # New bond dimensions per bond = sum over states
    new_bonds = [
        sum(bonds[b] for bonds in bond_dims_per_state)
        for b in range(num_bonds)
    ]

    # Use max bond dimension as the 'chi' attribute for the new state
    new_chi = max(new_bonds) if new_bonds else 1

    # Allocate new cores with per-bond chi's
    new_As = _construct_core_As(
        L=L,
        d=d,
        chi=new_bonds,
        out_dim=1,
        device=device,
        dtype=dtype,
    )

    # Compute per-state offsets along each bond (for block-diagonal embedding)
    offsets = [[0] * num_bonds for _ in range(num_states)]
    for b in range(num_bonds):
        cur = 0
        for s in range(num_states):
            offsets[s][b] = cur
            cur += bond_dims_per_state[s][b]

    # Fill new_As with blocks from each state
    for s, mp in enumerate(mpstates):
        As = mp.As
        bonds = bond_dims_per_state[s]
        offs = offsets[s]

        # Site 0: (d, chi_0)
        l = offs[0]
        r = l + bonds[0]
        new_As[0][:, l:r] = As[0]

        # Interior sites: 1 .. L-2
        for i in range(1, L - 1):
            left_bond = i - 1
            right_bond = i
            l0 = offs[left_bond]
            r0 = l0 + bonds[left_bond]
            l1 = offs[right_bond]
            r1 = l1 + bonds[right_bond]
            new_As[i][l0:r0, :, l1:r1] = As[i]

        # Last site: L-1, (chi_{L-2}, d)
        last_left_bond = num_bonds - 1  # = L-2
        l = offs[last_left_bond]
        r = l + bonds[last_left_bond]
        new_As[L - 1][l:r, :] = As[L - 1]

    new_mpstate = MPState(
        L=L,
        d=d,
        As=new_As,
        device=device,
        dtype=dtype,
    )
    return new_mpstate

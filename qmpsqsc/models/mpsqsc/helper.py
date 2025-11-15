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





# Helper functions for mpsqsc
def build_qsc_from_mpstate(mpstate: MPState) -> MpsQsc:
    """
    Build a MpsQsc from an MPState.
    """

    As = mpstate.As
    chi = mpstate.chi
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
    chi1 = mpstate1.chi
    d1 = mpstate1.d
    L1 = mpstate1.L

    As2 = mpstate2.As
    chi2 = mpstate2.chi
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


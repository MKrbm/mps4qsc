import torch
from typing import List
from ..mpstate import MPState
from .mpsqsc import MpsQsc


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
    As = _construct_core_As(L1, d1, chi, device=mpstate1.device, dtype=mpstate1.dtype)
    As[0][:, :chi1] = As1[0]
    As[0][:, chi1:] = As2[0]
    for i in range(1, L1 - 1):
        As[i][:chi1, :, :chi1] = As1[i]
        As[i][chi1:, :, chi1:] = As2[i]
    As[L1 - 1][:chi1, :, 0] = As1[L1 - 1]
    As[L1 - 1][chi1:, :, 1] = As2[L1 - 1]

    mpsqsc = MpsQsc(L1, chi, d1, As, device=mpstate1.device, dtype=mpstate1.dtype)
    return mpsqsc

def _construct_core_As(
    L: int,
    d: int,
    chi: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> List[torch.Tensor]:
    """ Return As with all zeros"""

    As: List[torch.Tensor] = []

    As.append(torch.zeros(d, chi, device=device, dtype=dtype))
    for _ in range(1, L - 1):
        As.append(torch.zeros(chi, d, chi, device=device, dtype=dtype))
    As.append(torch.zeros(chi, d, 2, device=device, dtype=dtype))
    return As
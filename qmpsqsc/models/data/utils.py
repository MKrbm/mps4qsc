import torch
from ..mpsqsc.mpstate import MPState
from typing import List

def flip_sites_in_mps(mps_state : MPState, site_indices: List[int]) -> MPState:
    """
    In-place flip of given sites in an MPS.

    Assumes:
        - mps_state[site] is a torch.Tensor
        - physical index is dimension 1 and has size 2
    """
    mps_state = mps_state.copy()
    As = mps_state.As
    L = len(As)
    X = torch.tensor([[0, 1], [1, 0]], device=As[0].device, dtype=As[0].dtype)
    for i in site_indices:
        A = mps_state.As[i]
        if i != 0 and i != L - 1:
            A = torch.einsum("iaj, ab -> ibj", A, X)
        elif i == 0:
            A = torch.einsum("aj, ab -> bj", A, X)
        else:  # ind == L - 1
            A = torch.einsum("ia, ab -> ib", A, X)
        mps_state.As[i] = A
    return mps_state
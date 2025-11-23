import torch
from ..mpsqsc.mpstate import MPState
from typing import List, Literal

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, 1j], [-1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

def flip_sites_in_mps(mps_state : MPState, site_indices: List[int], flip: Literal["X", "Y", "Z"] = "X") -> MPState:
    """
    In-place flip of given sites in an MPS.

    Assumes:
        - mps_state[site] is a torch.Tensor
        - physical index is dimension 1 and has size 2
    """
    _mps_state = mps_state.copy()
    As = _mps_state.As
    L = len(As)
    if flip == "X":
        F = X
    elif flip == "Y":
        F = Y
    elif flip == "Z":
        F = Z
    else:
        raise ValueError(f"Invalid flip: {flip}")
    for i in site_indices:
        A = _mps_state.As[i]
        if i != 0 and i != L - 1:
            A = torch.einsum("iaj, ab -> ibj", A, F)
        elif i == 0:
            A = torch.einsum("aj, ab -> bj", A, F)
        else:  # ind == L - 1
            A = torch.einsum("ia, ab -> ib", A, F)
        _mps_state.As[i] = A
    return _mps_state
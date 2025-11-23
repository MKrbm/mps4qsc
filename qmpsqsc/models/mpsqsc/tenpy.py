from tenpy.networks.site import BosonSite  # type: ignore
from tenpy.networks.mps import MPS as TenpyMPS  # type: ignore
from .mpstate import MPState
import numpy as np
import torch


def to_tenpy_mps(mpstate: MPState) -> TenpyMPS:
    d = mpstate.d
    L = mpstate.L
    site = BosonSite(Nmax=d - 1, conserve=None)
    sites = [site] * L

    # TenPy MPS.from_Bflat expects Bflat tensors with legs ('p', 'vL', 'vR'),
    # i.e. shape (d, chi_left, chi_right). :contentReference[oaicite:2]{index=2}
    Bflat: list[np.ndarray] = []
    As_cpu = mpstate.As
    for i, A in enumerate(As_cpu):
        # A: (chi_left, d, chi_right) -> B: (d, chi_left, chi_right)
        if i == 0:
            A = A.reshape(1, d, -1)
        elif i == L - 1:
            A = A.reshape(-1, d, 1)
        A_np = A.permute(1, 0, 2).contiguous().numpy()
        Bflat.append(A_np)

    psi = TenpyMPS.from_Bflat(
        sites,
        Bflat,
        SVs=None,
        bc="finite",
    )

    return psi

def tenpy_to_mpstate(psi: TenpyMPS) -> MPState:
    As = []
    dtype = psi.dtype
    dtype_torch = torch.complex128 if dtype == np.complex128 else torch.float64
    for i in range(len(psi.sites)):
        B = psi.get_B(i, form="A")
        As.append(torch.from_numpy(B.to_ndarray()).to(dtype_torch))
    
    d = psi.sites[0].Nmax + 1
    L = len(psi.sites)

    As[0] = As[0].reshape(d, -1)
    As[-1] = As[-1].reshape(-1, d)
    return MPState(L=L, d=d, As=As, dtype=dtype_torch)
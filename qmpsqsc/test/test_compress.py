# test_mpstate_compression.py

import math
import torch

# Adjust these imports to match your project structure
# from .mpstate import MPState
# from .compression import variational_refine, adam_refine, compress_mpstate

from qmpsqsc.models.mpsqsc.mpstate import MPState
from qmpsqsc.models.mpsqsc.compress import compress_mpstate, adam_refine



def _fidelity(psi: MPState, phi: MPState) -> float:
    """
    Compute normalized fidelity F = |<psi|phi>|^2 / (||psi||^2 ||phi||^2) as a float.
    """
    ov = psi.inner_product(phi)  # complex scalar ~ <phi|psi>
    npsi2 = torch.real(psi.inner_product(psi))
    nphi2 = torch.real(phi.inner_product(phi))

    npsi = torch.sqrt(npsi2 + 1e-12)
    nphi = torch.sqrt(nphi2 + 1e-12)

    fid = (ov.abs() / (npsi * nphi)) ** 2
    return float(fid.item())


def _max_bond_dim(mps: MPState) -> int:
    """
    Max internal bond dimension for an MPS with
    - A[0] shape  (d, chi_0)
    - A[i] shape  (chi_{i-1}, d, chi_i)
    - A[L-1] shape (chi_{L-1}, d)
    """
    max_chi = 1
    L = mps.L

    for i, A in enumerate(mps.As):
        if A.ndim == 3:
            chi_l, _, chi_r = A.shape
        elif A.ndim == 2:
            if i == 0:
                chi_l, chi_r = 1, A.shape[1]
            elif i == L - 1:
                chi_l, chi_r = A.shape[0], 1
            else:
                raise ValueError(f"Unexpected 2D tensor at non-boundary site {i}")
        else:
            raise ValueError(f"Unexpected ndim={A.ndim} at site {i}")

        max_chi = max(max_chi, chi_l, chi_r)

    return max_chi



def test_compress_mpstate_improves_or_matches_truncation():
    """
    Check that variational_refine(psi, psi.truncate(...)) does not decrease fidelity
    (up to tiny numerical noise).
    """
    torch.manual_seed(0)

    L = 10
    d = 2
    chi_big = 16
    chi_small = 8

    psi = MPState(L=L, d=d, chi=chi_big, init="random", seed=123)
    psi.normalize(inplace=True)

    # NOTE: adapt this to your actual method name/signature
    # e.g. psi.trunctate(chi_small) if that's what you have.
    phi0 = psi.truncate_bond_dimension(chi_small)
    phi0.normalize(inplace=True)

    fid0 = _fidelity(psi, phi0)
    phi1, fid1 = compress_mpstate(psi, target_chi=chi_small, n_sweeps=10, adam_steps=10)
    fid1 = _fidelity(psi, phi1)

    # Basic sanity: bond dimension preserved
    assert _max_bond_dim(phi1) <= chi_small

    # Allow tiny numerical degradation, but no real drop
    assert fid1 >= fid0 - 1e-5, f"variational_refine decreased fidelity: {fid1} < {fid0}"


def test_adam_refine_does_not_make_things_worse():
    """
    Check that Adam-based refinement starting from a truncated state
    at least does not seriously hurt fidelity.
    """
    torch.manual_seed(1)

    L = 8
    d = 2
    chi_big = 12
    chi_small = 8

    psi = MPState(L=L, d=d, chi=chi_big, init="random", seed=321)
    phi0 = psi.truncate_bond_dimension(chi_small)
    phi0.normalize(inplace=True)

    fid0 = _fidelity(psi, phi0)

    # Short-ish optimization; you can bump steps if you want stronger guarantees
    phi_adam = adam_refine(
        psi,
        phi0,
        steps=50,
        lr=1e-3,
        normalize_each_step=True,
        verbose=False,
    )
    fid1 = _fidelity(psi, phi_adam)

    assert _max_bond_dim(phi_adam) <= chi_small

    # Adam might occasionally plateau, but shouldn't tank fidelity.
    # Allow a small numerical slack.
    assert fid1 >= fid0 - 1e-4, f"adam_refine decreased fidelity: {fid1} < {fid0}"



import opt_einsum as oe # type: ignore[import-untyped]
import torch
from typing import List
import math

from qmpsqsc.models.mpsqsc import MpsQsc, MPState
from qmpsqsc.models.qmps.utils import embed_iso, construct_unitary_from_As
from qmpsqsc.models.qmps.qmps import qMPS


def gen_iso_matrix(N: int, M: int) -> torch.Tensor:
    """
    Generate a random isometry matrix.
    """
    G = max(N, M)
    H = torch.randn(G, G, dtype=torch.complex128)
    H = H + H.mH
    E, V = torch.linalg.eigh(H)
    V = V[:N, :M]
    return V


def test_gen_iso_matrix():
    """
    Check that the embedding is correct.
    """
    N = 5
    M = 3
    V = gen_iso_matrix(N, M)
    assert torch.allclose(V.mH @ V, torch.eye(M, dtype=V.dtype))
    
def test_gen_iso_matrix_N_less_than_M():
    """
    Check that the embedding is correct.
    """
    N = 3
    M = 5
    V = gen_iso_matrix(N, M)
    assert torch.allclose(V @ V.mH, torch.eye(N, dtype=V.dtype))

def test_embed_iso_unitary_valid():
    """
    Check that the embedding is correct.
    """
    M = 3
    N = 5
    V = gen_iso_matrix(M, N)
    U = embed_iso(V)
    assert torch.allclose(U @ U.mH, torch.eye(N, dtype=U.dtype))
    

def test_embed_iso():
    """
    Check that the embedding is correct.
    """
    M = 3
    N = 6
    V = gen_iso_matrix(M, N)
    U = embed_iso(V).reshape(2, M, N)
    VU = U[0, :, :].contiguous()
    assert torch.allclose(V, VU)

def test_construct_unitary_from_mpsqsc():
    """
    Check that the unitary is constructed correctly.
    """
    L = 5
    chi = 4
    d = 2
    mpsqsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.complex128)
    mpsqsc = mpsqsc.canonicalize(truncate=True, normalize=True)
    As = mpsqsc.As

    Us, last = construct_unitary_from_As(As, merge_first_two=False)

    U0 = Us[0]
    assert torch.allclose(U0 @ U0.mH, torch.eye(U0.shape[1], dtype=U0.dtype))
    assert torch.allclose(U0, As[0].T)

    U1 = Us[1]
    assert torch.allclose(U1 @ U1.mH, torch.eye(U1.shape[1], dtype=U1.dtype))
    assert torch.allclose(U1, As[1].reshape(-1, As[1].shape[-1]).mT)

    for i in range(2, L-1):
        Ui = Us[i]
        Ai = As[i].reshape(-1, As[i].shape[-1]).T

        assert torch.allclose(Ui @ Ui.mH, torch.eye(Ui.shape[1], dtype=Ui.dtype))
        Ui = Ui.reshape(d, chi, -1)[0]
        assert torch.allclose(Ui, Ai)
    

    UL = Us[L-1]
    assert torch.allclose(UL @ UL.mH, torch.eye(UL.shape[1], dtype=UL.dtype))

    UL = UL.reshape(-1, 2, d * chi)
    last = last.reshape(2, 2, 2, 2)
    
    U_last = torch.einsum("aci, dlbc -> adlbi", UL, last)
    last_ps = U_last[0, 0, :, 0, :]
    As_last = As[L-1].reshape(-1, 2).mT

    norm1 = torch.linalg.norm(last_ps)
    norm2 = torch.linalg.norm(As_last)
    scale = norm1 / norm2

    assert torch.allclose(last_ps, As_last * scale) # scaled the last core 

def test_contract_circuit_with_state_partial_trace_is_density_matrix():
    """
    Check that the partial trace is correct.
    """
    L = 5
    chi = 4
    chi_state = 2
    d = 2
    mpsqsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.complex128)
    mpsqsc = mpsqsc.canonicalize(truncate=True, normalize=True)
    Us, last = construct_unitary_from_As(mpsqsc.As, merge_first_two=True)
    qmpsqsc = qMPS(L=L, chi=chi, d=d, Us=Us, last_unitary=last, seed=42)
    mpstate = MPState(L=L, chi=chi_state, d=d, init="random", seed=42, dtype=torch.complex128)

    vals = qmpsqsc._contract_circuit_with_state_partial_trace(mpstate)
    # check if vlas is a density matrix
    #  positivity check
    assert torch.allclose(vals, vals.mH)
    E, V = torch.linalg.eigh(vals)
    assert torch.all(E >= 0)

    # trace check
    norm = mpstate.norm()
    assert torch.allclose(torch.trace(vals), torch.tensor(norm**2, dtype=vals.dtype))


def test_contract_circuit_with_state_ae_same_as_mpsqsc():
    """
    Check that the partial trace is correct.
    """
    L = 5
    chi = 4
    chi_state = 2
    d = 2
    mpsqsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.complex128)
    mpsqsc = mpsqsc.canonicalize(truncate=True, normalize=True)
    Us, last = construct_unitary_from_As(mpsqsc.As, merge_first_two=True)
    qmpsqsc = qMPS(L=L, chi=chi, d=d, Us=Us, last_unitary=last, seed=42)
    mpstate = MPState(L=L, chi=chi_state, d=d, init="random", seed=42, dtype=torch.complex128)

    vals_ae = qmpsqsc._contract_circuit_with_state_ae(mpstate)
    pred_mps = mpsqsc.contract_with_state(mpstate)

    # extract diagonal parts of vals_ae
    pred_ae = vals_ae.real.diagonal(dim1=0, dim2=1)
    # normalize so that summation is 1
    pred_ae = pred_ae / torch.sum(pred_ae)
    # normalize pred_mps so that norm is 1
    pred_mps = pred_mps / torch.linalg.norm(pred_mps)
    pred_mps = torch.abs(pred_mps) ** 2

    assert torch.allclose(pred_ae, pred_mps)


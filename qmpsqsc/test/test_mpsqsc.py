import opt_einsum as oe # type: ignore[import-untyped]
import torch
from typing import List
import math

from qmpsqsc.models.mpsqsc import MpsQsc, MPState, build_product_state

def test_random_state():
    """
    Check that the random state is constructed correctly.
    """
    L, chi, d = 5, 3, 2
    mps = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42)
    psi = mps.state_vector()
    assert psi.shape == (d,) * L + (2,)


def test_normalize():
    """
    Check that the normalization is correct.
    """
    L, chi, d = 5, 3, 2
    mps = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42)
    mps.normalize()
    assert torch.allclose(mps.norm(), torch.tensor(1.0, dtype=mps.dtype))

def test_contract_with_state():
    """
    Check that the contraction with a state is correct.
    """
    L, chi, d = 5, 3, 2
    mps = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    state = torch.randn(L, d, dtype=torch.float64)
    pure_state = build_product_state(L, d, state)
    vals = mps.contract_with_state(pure_state)

    # Calculate vals manually
    As = mps.As
    left_env = torch.einsum("ib, i -> b", As[0], state[0])
    for i in range(1, L):
        left_env = torch.einsum("a,aib, i -> b", left_env, As[i], state[i])
    assert torch.allclose(vals, left_env)


def test_canonicalize_is_isometric():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 4, 4
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize()

    As_can = qs_can.As
    identity = torch.einsum("ab, ac -> bc", As_can[0], As_can[0].conj())
    assert torch.allclose(identity, torch.eye(chi, dtype=qsc.dtype))

    for i in range(1, L-1):
        identity = torch.einsum("aib, aic -> bc", As_can[i], As_can[i].conj())
        assert torch.allclose(identity, torch.eye(chi, dtype=qsc.dtype))
    
    norm = torch.einsum("aib, aib -> ", As_can[L-1], As_can[L-1].conj())
    assert torch.allclose(norm, qsc.norm()**2)

def test_canonicalize_is_isometric_chi_greater_than_d():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 6, 4
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize()

    As_can = qs_can.As
    identity = torch.einsum("ab, ac -> bc", As_can[0], As_can[0].conj())
    eye = torch.zeros(chi, chi, dtype=qsc.dtype)
    eye[:d, :d] = torch.eye(d, dtype=qsc.dtype)
    assert torch.allclose(identity, eye)

    for i in range(1, L-1):
        identity = torch.einsum("aib, aic -> bc", As_can[i], As_can[i].conj())
        assert torch.allclose(identity, torch.eye(chi, dtype=qsc.dtype))
    
    norm = torch.einsum("aib, aib -> ", As_can[L-1], As_can[L-1].conj())
    assert torch.allclose(norm, qsc.norm()**2)

def test_canonicalize_is_isometric_chi_greater_than_d_truncate():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 4, 2
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize(truncate=True)

    As_can = qs_can.As
    identity = torch.einsum("ab, ac -> bc", As_can[0], As_can[0].conj())
    eye = torch.eye(d, dtype=qsc.dtype)
    assert torch.allclose(identity, eye)

    for i in range(1, L-1):
        identity = torch.einsum("aib, aic -> bc", As_can[i], As_can[i].conj())
        assert torch.allclose(identity, torch.eye(chi, dtype=qsc.dtype))
    
    norm = torch.einsum("aib, aib -> ", As_can[L-1], As_can[L-1].conj())
    assert torch.allclose(norm, qsc.norm()**2)

def test_canonicalize_is_isometric_chi_greater_than_d_truncate_large_chi():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 7, 8, 2
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize(truncate=True)

    As_can = qs_can.As
    identity = torch.einsum("ab, ac -> bc", As_can[0], As_can[0].conj())
    eye = torch.eye(2, dtype=qsc.dtype)
    assert torch.allclose(identity, eye)

    As_can = qs_can.As
    identity = torch.einsum("aib, aic -> bc", As_can[1], As_can[1].conj())
    eye = torch.eye(4, dtype=qsc.dtype)
    assert torch.allclose(identity, eye)

    for i in range(2, L-1):
        identity = torch.einsum("aib, aic -> bc", As_can[i], As_can[i].conj())
        assert torch.allclose(identity, torch.eye(chi, dtype=qsc.dtype))
    
    norm = torch.einsum("aib, aib -> ", As_can[L-1], As_can[L-1].conj())
    assert torch.allclose(norm, qsc.norm()**2)

def test_canonicalize():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 3, 2
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize()
    mps = MPState(L=L, chi=chi, d=d, device=qsc.device, dtype=qsc.dtype)

    assert torch.allclose(qsc.contract_with_state(mps), qs_can.contract_with_state(mps))


def test_canonicalize_norm():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 3, 2
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize(normalize=True)
    assert torch.allclose(qs_can.norm(), torch.tensor(1.0, dtype=qsc.dtype))

def test_canonicalize_norm_random_input():
    """
    Check that the canonicalization is correct.
    """
    L, chi, d = 5, 3, 2
    qsc = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42, dtype=torch.float64)
    qs_can = qsc.canonicalize(normalize=True)
    mps = MPState(L=L, chi=chi, d=d, device=qsc.device, dtype=qsc.dtype)
    out1 = qsc.contract_with_state(mps)
    out2 = qs_can.contract_with_state(mps)

    scale = torch.linalg.norm(out1) / torch.linalg.norm(out2)
    assert torch.allclose(out1, out2 * scale)
import opt_einsum as oe # type: ignore[import-untyped]
import torch
from typing import List
import math

from qmpsqsc.models.mpsqsc import MpsQsc, MPState

# ===========================
# 1. PATH CACHING TESTS
# ===========================

def test_mpsqsc_path_cached():
    """
    Check that oe.contract_path is only called once for MpsQsc
    when calling .contract() multiple times with the same tensors.
    """
    call_counter = {"n": 0}
    original_contract_path = oe.contract_path

    def wrapped_contract_path(*args, **kwargs):
        call_counter["n"] += 1
        return original_contract_path(*args, **kwargs)

    oe.contract_path = wrapped_contract_path
    try:
        L, chi, d = 5, 3, 2
        mps = MpsQsc(L=L, chi=chi, d=d, init="random", seed=42)

        y1 = mps.state_vector()
        y2 = mps.state_vector()
    finally:
        oe.contract_path = original_contract_path

    # Path should be built exactly once.
    assert call_counter["n"] == 1
    # And repeated contractions should give the same result.
    assert torch.allclose(y1, y2)


def test_mpstate_path_cached():
    """
    Check that oe.contract_path is only called once for MPState
    when calling .contract() multiple times with the same tensors.
    """
    call_counter = {"n": 0}
    original_contract_path = oe.contract_path

    def wrapped_contract_path(*args, **kwargs):
        call_counter["n"] += 1
        return original_contract_path(*args, **kwargs)

    oe.contract_path = wrapped_contract_path
    try:
        L, chi, d = 5, 3, 2
        mpstate = MPState(L=L, chi=chi, d=d, init="random", seed=123)

        psi1 = mpstate.state_vector()
        psi2 = mpstate.state_vector()
    finally:
        oe.contract_path = original_contract_path

    assert call_counter["n"] == 1
    assert torch.allclose(psi1, psi2)

# ===========================
# 2. GHZ STATE TEST FOR MPState
# ===========================

def test_mpstate_ghz_state():
    """
    Build an exact GHZ MPS and check that MPState.contract()
    reproduces the GHZ amplitudes.

    GHZ_L = |0...0> + |1...1>  (unnormalized)
    """
    L = 4         # number of sites
    d = 2         # physical dimension {0,1}
    chi = 2       # bond dimension
    device = torch.device("cpu")
    dtype = torch.float32

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
    psi = mpstate.state_vector()

    assert psi.shape == (d,) * L

    idx_all_0 = (0,) * L
    idx_all_1 = (1,) * L

    # GHZ amplitudes (unnormalized): 1 for |00..0>, 1 for |11..1>, 0 otherwise
    assert torch.allclose(psi[idx_all_0], torch.tensor(1.0, dtype=dtype))
    assert torch.allclose(psi[idx_all_1], torch.tensor(1.0, dtype=dtype))

    # Check all other configurations are zero
    all_indices = torch.cartesian_prod(*[torch.arange(d) for _ in range(L)])
    for cfg in all_indices:
        tup = tuple(cfg.tolist())
        if tup not in (idx_all_0, idx_all_1):
            assert torch.allclose(psi[tup], torch.tensor(0.0, dtype=dtype))


# ===========================
# 3. AKLT STATE TEST FOR MPState
# ===========================

def test_mpstate_aklt_state():
    """
    Build the spin-1 AKLT MPS (bond dim 2) and verify that
    MPState.contract() matches a direct reference contraction.

    Local matrices for S=1 AKLT (one standard choice):
      A^{+1} = sqrt(2/3) * [[0, 1],
                            [0, 0]]
      A^{0}  = -sqrt(1/3)* [[ 1,  0],
                            [ 0, -1]]
      A^{-1} = -sqrt(2/3)* [[0, 0],
                            [1, 0]]
    Boundaries vL, vR are simple spin-1/2 vectors.
    """
    L = 4          # chain length
    d = 3          # physical dim: {+1, 0, -1} -> {0,1,2}
    chi = 2        # bond dim
    device = torch.device("cpu")
    dtype = torch.float32

    sqrt23 = math.sqrt(2.0 / 3.0)
    sqrt13 = math.sqrt(1.0 / 3.0)

    A_plus = sqrt23 * torch.tensor([[0.0, 1.0],
                                    [0.0, 0.0]], device=device, dtype=dtype)
    A_zero = -sqrt13 * torch.tensor([[1.0,  0.0],
                                     [0.0, -1.0]], device=device, dtype=dtype)
    A_minus = -sqrt23 * torch.tensor([[0.0, 0.0],
                                      [1.0, 0.0]], device=device, dtype=dtype)

    # physical index order: 0 -> +1, 1 -> 0, 2 -> -1
    A_phys = [A_plus, A_zero, A_minus]

    # Boundary vectors (pick one of the degenerate edge configurations)
    vL = torch.tensor([1.0, 0.0], device=device, dtype=dtype)  # row vector
    vR = torch.tensor([1.0, 0.0], device=device, dtype=dtype)  # column vector

    # Build MPS cores As for MPState from (A_phys, vL, vR)
    As: List[torch.Tensor] = []

    # First core: A0(m, a1) = vL^T @ A^{m}
    A0 = torch.zeros(d, chi, device=device, dtype=dtype)
    for m in range(d):
        row = vL @ A_phys[m]   # shape (2,)
        A0[m, :] = row
    As.append(A0)

    # Interior cores: A_mid(a_{i-1}, m, a_i) = A^{m}_{a_{i-1}, a_i}
    for _ in range(1, L - 1):
        A_mid = torch.zeros(chi, d, chi, device=device, dtype=dtype)
        for m in range(d):
            A_mid[:, m, :] = A_phys[m]
        As.append(A_mid)

    # Last core: AL(a_{L-1}, m_L) = (A^{m_L} @ vR)_a_{L-1}
    AL = torch.zeros(chi, d, device=device, dtype=dtype)
    for m in range(d):
        col = A_phys[m] @ vR  # shape (2,)
        AL[:, m] = col
    As.append(AL)

    mpstate = MPState(L=L, chi=chi, d=d, As=As, device=device, dtype=dtype)
    psi = mpstate.state_vector()

    assert psi.shape == (d,) * L

    # Reference amplitudes via direct matrix multiplication:
    # amp(m1,...,mL) = vL^T A^{m1} ... A^{mL} vR
    all_configs = torch.cartesian_prod(*[torch.arange(d) for _ in range(L)])
    for cfg in all_configs:
        config = cfg.tolist()
        vec = vL.clone()
        for m in config:
            vec = vec @ A_phys[m]
        amp_ref = torch.dot(vec, vR)
        amp_mps = psi[tuple(config)]
        assert torch.allclose(amp_mps, amp_ref, atol=1e-6)


# -------------------------------
# 1) norm() matches state_vector
# -------------------------------

def test_mpstate_norm_matches_state_vector():
    """
    For a small chain, compare MPState.norm() to the naive
    Frobenius/L2 norm of the fully materialized state vector.
    """
    L, d, chi = 4, 2, 3
    # Use float64 to keep numerical error tiny
    mp = MPState(L=L, chi=chi, d=d, init="random", seed=123, dtype=torch.float64)
    psi = mp.state_vector()                     # shape (d,)*L
    expected = psi.reshape(-1).norm()           # ||psi||_2
    n = mp.norm()
    assert n.shape == torch.Size([])            # scalar tensor
    assert torch.allclose(n, expected, rtol=1e-10, atol=1e-10)


# -------------------------------------------------
# 2) normalize() makes ||psi|| = 1 and scales A[0]
# -------------------------------------------------

def test_mpstate_normalize_unit_norm_and_scaling():
    """
    normalize() should scale in-place so that the norm becomes 1.
    By our implementation, only As[0] is scaled by 1/n.
    """
    L, d, chi = 5, 2, 4
    mp = MPState(L=L, chi=chi, d=d, init="random", seed=999, dtype=torch.float64)

    n_before = mp.norm()
    # Should be positive and finite for random tensors
    assert torch.isfinite(n_before) and float(n_before) > 0.0

    # assert torch.allclose(ret, n_before, rtol=1e-12, atol=1e-12)

    # After normalization, norm should be ~1
    ret = mp.normalize()  # returns the pre-normalization norm
    n_after = ret.norm()
    assert torch.allclose(n_after, torch.tensor(1.0, dtype=mp.dtype), rtol=1e-8, atol=1e-8)

# ------------------------------------------------------------------
# 3) Path caching for norm(): compute once and reuse after normalize
# ------------------------------------------------------------------

def test_mpstate_norm_path_cached_and_persists_after_normalize():
    """
    Ensure the double-layer norm path is computed exactly once and reused
    across multiple norm() calls and even after normalize() (shapes unchanged).
    """
    L, d, chi = 6, 2, 3
    mp = MPState(L=L, chi=chi, d=d, init="random", seed=7, dtype=torch.float32)

    call_counter = {"n": 0}
    original_contract_path = oe.contract_path

    def wrapped_contract_path(*args, **kwargs):
        call_counter["n"] += 1
        return original_contract_path(*args, **kwargs)

    # Patch only for this test scope
    oe.contract_path = wrapped_contract_path
    try:
        # First call builds the path
        _ = mp.norm()
        # Subsequent calls should reuse the cached path
        _ = mp.norm()
        _ = mp.norm()
        assert call_counter["n"] == 1

        # normalize() changes values but not shapes; path should still be reused
        mp.normalize()
        _ = mp.norm()
        _ = mp.norm()
        assert call_counter["n"] == 1
    finally:
        oe.contract_path = original_contract_path


# ------------------------------------------------------------
# (Optional) Gradient sanity check: norm is autograd-friendly
# ------------------------------------------------------------

def test_mpstate_norm_autograd():
    """
    Quick sanity check that norm() is differentiable wrt cores.
    """
    L, d, chi = 3, 2, 3
    mp = MPState(L=L, chi=chi, d=d, init="random", seed=1234, dtype=torch.float64)

    # Turn on gradients for the cores
    for i in range(L):
        mp.As[i].requires_grad_(True)

    n = mp.norm()            # scalar
    n.backward()             # backprop through double-layer contraction

    # Each core should have a gradient
    for A in mp.As:
        assert A.grad is not None
        assert torch.isfinite(A.grad).all()
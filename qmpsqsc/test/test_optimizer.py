import math
import torch
import pytest
from qmpsqsc.models.qmps.optimizer import matrix_root, matrix_root_inv, cayley, StiefelAdam

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

### Compute matrix root inversion by eigen-decomposition. Super expensive. For debug only.
def mat_root_inv_for_debug(A):
    # Force Hermitian (for numerical issues) – works for real or complex
    A_herm = 0.5 * (A + A.mH)
    D, U = torch.linalg.eigh(A_herm)
    D = D.to(A.dtype)
    Dinv = torch.diag(1.0 / torch.sqrt(D))
    return U @ Dinv @ U.mH

def random_orthogonal(n, dtype=torch.float64, device="cpu"):
    """Generate a random orthogonal (or real unitary) matrix Q (Q^T Q = I)."""
    M = torch.randn(n, n, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(M)
    # Fix sign for determinism (optional)
    diag = torch.sign(torch.diagonal(R))
    Q = Q * diag
    return Q


def random_unitary(n, dtype=torch.complex128, device="cpu"):
    """Generate a random unitary matrix U (U^* U = I)."""
    M = torch.randn(n, n, dtype=torch.float64, device=device)
    N = torch.randn(n, n, dtype=torch.float64, device=device)
    Z = M + 1j * N
    Q, R = torch.linalg.qr(Z)
    # Normalize diagonal of R to have unit magnitude
    diag = torch.diagonal(R)
    phase = diag / torch.abs(diag)
    Q = Q * phase
    return Q.to(dtype)


def make_spd_matrix(n, dtype=torch.float64, device="cpu"):
    """Make a symmetric/Hermitian positive definite matrix."""
    if dtype.is_complex:
        A_real = torch.randn(n, n, dtype=torch.float64, device=device)
        A_imag = torch.randn(n, n, dtype=torch.float64, device=device)
        M = A_real + 1j * A_imag
        A = M.mH @ M
    else:
        M = torch.randn(n, n, dtype=dtype, device=device)
        A = M.T @ M
    # Add a bit to the diagonal to push eigenvalues away from 0.
    A = A + 1e-1 * torch.eye(n, dtype=A.dtype, device=A.device)
    # Force Hermitian/symmetric
    A = 0.5 * (A + A.mH)
    return A


def fro_norm(X):
    return torch.linalg.norm(X, ord="fro")


def loss_quadratic(X, A):
    """Simple loss f(X) = 0.5 ||X - A||_F^2."""
    return 0.5 * fro_norm(X - A) ** 2


def random_skew_hermitian(n, dtype, device="cpu"):
    """Random skew-Hermitian matrix K (K^* = -K)."""
    if dtype.is_complex:
        R = torch.randn(n, n, dtype=torch.float64, device=device)
        I = torch.randn(n, n, dtype=torch.float64, device=device)
        Z = R + 1j * I
        H = 0.5 * (Z + Z.mH)  # Hermitian
    else:
        Z = torch.randn(n, n, dtype=dtype, device=device)
        H = 0.5 * (Z + Z.T)  # symmetric (real Hermitian)
    K = 1j * H if dtype.is_complex else (H - H.T)
    return K.to(dtype)


# ---------------------------------------------------------------------
# 0. Orthogonal matrix treated as complex vs float
# ---------------------------------------------------------------------

def test_orthogonal_complex_equivalence_matrix_root_inv():
    torch.manual_seed(0)
    n = 6

    Q_real = random_orthogonal(n, dtype=torch.float64)
    Q_complex = Q_real.to(torch.complex128)

    # Gram matrices (both are identity theoretically)
    G_real = Q_real.T @ Q_real       # float64, symmetric
    G_complex = Q_complex.mH @ Q_complex  # complex128, Hermitian

    Rinv_real = matrix_root_inv(G_real)       # from your implementation
    Rinv_complex = matrix_root_inv(G_complex) # same function, complex path

    # Complex result should be (numerically) real and equal to Rinv_real
    assert torch.allclose(Rinv_real, Rinv_complex.real, atol=1e-6, rtol=1e-6)
    assert torch.max(torch.abs(Rinv_complex.imag)) < 1e-7


# ---------------------------------------------------------------------
# 1. Check sqrt / inv-sqrt correctness for float and complex
# ---------------------------------------------------------------------

def _check_sqrt_inv_for_dtype(dtype):
    torch.manual_seed(1)
    n = 5
    A = make_spd_matrix(n, dtype=dtype)

    A_root = matrix_root(A)
    A_root_inv = matrix_root_inv(A)

    A_recon = A_root @ A_root
    I_recon = A_root_inv @ A_root_inv @ A

    eye = torch.eye(n, dtype=dtype, device=A.device)

    # A_root^2 ≈ A
    assert torch.allclose(A_recon, A, atol=1e-5, rtol=1e-5)

    # A^{-1/2} A A^{-1/2} ≈ I
    assert torch.allclose(I_recon, eye, atol=1e-5, rtol=1e-5)

    # Debug version via eigen-decomposition should match inverse sqrt
    A_root_inv_debug = mat_root_inv_for_debug(A)
    assert torch.allclose(A_root_inv, A_root_inv_debug, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_sqrt_inv_real_and_complex(dtype):
    _check_sqrt_inv_for_dtype(dtype)


# ---------------------------------------------------------------------
# 2. Gradient is steepest descent direction (Lie algebra norm 1)
# ---------------------------------------------------------------------

def _check_steepest_descent(dtype):
    torch.manual_seed(2)
    device = "cpu"
    n = 4

    # Work on the unitary/orthogonal group (square case)
    X = random_unitary(n, dtype=dtype, device=device) if dtype.is_complex \
        else random_orthogonal(n, dtype=dtype, device=device)

    A_target = torch.randn(n, n, dtype=dtype, device=device)
    if dtype.is_complex:
        A_target = A_target + 1j * torch.randn(n, n, dtype=dtype, device=device)

    def f(X):
        return loss_quadratic(X, A_target)

    # Euclidean gradient
    X.requires_grad_(True)
    loss = f(X)
    loss.backward()
    G = X.grad.detach()         # Euclidean gradient d f / dX
    X = X.detach()

    # Lie algebra gradient at X for canonical metric on U(n)/O(n):
    # grad_R = X @ skew(X^* G).
    XtG = X.mH @ G
    grad_lie = 0.5 * (XtG - XtG.mH)   # skew-Hermitian/s skew-symmetric

    if fro_norm(grad_lie) < 1e-12:
        # Degenerate case; just skip
        return

    # Normalize to unit Frobenius norm -> direction of steepest descent on Lie algebra
    K_grad = grad_lie / fro_norm(grad_lie)

    def manifold_step(X, K, t):
        # Move along geodesic: X -> X exp(-t K)
        return X @ torch.matrix_exp(-t * K)

    t = 1e-4
    base_loss = f(X).item()
    X_grad_step = manifold_step(X, K_grad, t)
    loss_grad = f(X_grad_step).item()
    delta_grad = (loss_grad - base_loss) / t   # directional derivative along -K_grad

    # Compare with many random tangent directions of same norm
    num_trials = 32
    best_delta = math.inf
    for _ in range(num_trials):
        K_rand = random_skew_hermitian(n, dtype=dtype, device=device)
        # Project to unit Fro norm
        K_rand = K_rand / fro_norm(K_rand)

        X_rand_step = manifold_step(X, K_rand, t)
        loss_rand = f(X_rand_step).item()
        delta_rand = (loss_rand - base_loss) / t

        best_delta = min(best_delta, delta_rand)

    # For steepest descent, delta_grad should be <= any other delta_rand (more negative),
    # up to numerical noise.
    assert delta_grad <= best_delta + 1e-4, (
        f"Gradient direction not steepest: delta_grad={delta_grad}, "
        f"best_random={best_delta}"
    )

    # And gradient direction should actually decrease the loss
    assert delta_grad < -1e-6, f"Gradient direction does not decrease loss: {delta_grad}"


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_steepest_descent_real_and_complex(dtype):
    _check_steepest_descent(dtype)


# ---------------------------------------------------------------------
# 3. Check loss decreases when optimizing orthogonal / unitary with StiefelAdam
# ---------------------------------------------------------------------
def _run_stiefel_adam_minimization(dtype):
    torch.manual_seed(3)
    device = "cpu"
    n = 6  # square for simplicity; you can use rectangular too.

    # Parameter to optimize: shape (n, n) living on O(n) / U(n).
    if dtype.is_complex:
        X0 = random_unitary(n, dtype=dtype, device=device)
    else:
        X0 = random_orthogonal(n, dtype=dtype, device=device)

    # We'll let the optimizer fix orthonormality if it's not exact.
    X = X0.clone().detach().requires_grad_(True)

    # Target unitary matrix for quadratic loss
    U_target = random_unitary(n, dtype=dtype, device=device)

    opt = StiefelAdam([X], lr=0.005, betas=(0, 0),
                      expm_method="MatrixExp", inner_prod="Canonical", inner_iter=8)

    def f():
        return loss_quadratic(X, U_target)

    with torch.enable_grad():
        for _ in range(20):
            opt.zero_grad()
            loss_val = f()
            loss_val.backward()
            opt.step()
            loss_val_current = f().item()
            assert loss_val_current < loss_val.item(), f"Loss did not decrease: {loss_val_current} < {loss_val.item()} for iteration {_}"
            # print(f"Loss: {loss_val_current}, Iteration: {_}")


    # Loss decreases overall
    # assert loss_hist[-1] < loss_hist[0] * 0.8, (
    #     f"Loss did not decrease enough: start={loss_hist[0]}, end={loss_hist[-1]}"
    # )

    # Check (approx) orthonormality at the end: X^* X ≈ I
    with torch.no_grad():
        Gram = X.detach().mH @ X.detach()
        I = torch.eye(n, dtype=dtype, device=device)
        assert torch.allclose(Gram, I, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_stiefel_adam_decreases_loss_real_and_complex(dtype):
    _run_stiefel_adam_minimization(dtype)


# # ---------------------------------------------------------------------
# # 4. Misc sanity checks
# #    - matrix_root_inv is Hermitian for Hermitian input
# #    - projection via matrix_root_inv actually improves orthonormality
# # ---------------------------------------------------------------------

def test_matrix_root_inv_hermitian_output():
    torch.manual_seed(4)
    n = 5
    A = make_spd_matrix(n, dtype=torch.complex128)
    Rinv = matrix_root_inv(A)

    # Rinv should be (approximately) Hermitian
    diff = Rinv - Rinv.mH
    assert fro_norm(diff) < 1e-6


def test_projection_improves_orthonormality():
    torch.manual_seed(5)
    n, m = 8, 5
    X = torch.randn(n, m, dtype=torch.float64)

    # Initial deviation from orthonormality
    Gram0 = X.T @ X
    I = torch.eye(m, dtype=torch.float64)
    dev0 = fro_norm(Gram0 - I).item()

    # Project via inverse square root of Gram: X <- X (X^T X)^{-1/2}
    Gram = X.T @ X
    Rinv = matrix_root_inv(Gram)
    X_proj = X @ Rinv

    Gram1 = X_proj.T @ X_proj
    dev1 = fro_norm(Gram1 - I).item()

    assert dev1 < dev0, f"Projection did not improve orthonormality: dev0={dev0}, dev1={dev1}"
    assert dev1 < 1e-6, f"Projected matrix not sufficiently orthonormal: dev1={dev1}"

@pytest.mark.parametrize("expm_method", ["MatrixExp"])
def test_stiefel_adam_real_vs_complex_same_start(expm_method):
    torch.manual_seed(7)
    device = "cpu"
    n = 6

    # Same real orthogonal initial point
    X0_real = random_orthogonal(n, dtype=torch.float64, device=device)

    # Same real orthogonal target
    U_target_real = random_orthogonal(n, dtype=torch.float64, device=device)

    # Real path
    X_real = X0_real.clone().detach().requires_grad_(True)
    U_real = U_target_real

    opt_real = StiefelAdam(
        [X_real],
        lr=0.01,
        betas=(0.9, 0.99),
        expm_method=expm_method,
        inner_prod="Canonical",
        inner_iter=8,
    )

    # Complex path: same matrices but cast to complex128
    X_cplx = X0_real.to(torch.complex128).clone().detach().requires_grad_(True)
    U_cplx = U_target_real.to(torch.complex128)

    opt_cplx = StiefelAdam(
        [X_cplx],
        lr=0.01,
        betas=(0.9, 0.99),
        expm_method=expm_method,
        inner_prod="Canonical",
        inner_iter=8,
    )

    def f_real():
        return loss_quadratic(X_real, U_real)

    def f_cplx():
        return loss_quadratic(X_cplx, U_cplx)

    num_steps = 50
    with torch.enable_grad():
        for _ in range(num_steps):
            # real step
            opt_real.zero_grad()
            loss_r = f_real()
            loss_r.backward()
            opt_real.step()

            # complex step
            opt_cplx.zero_grad()
            loss_c = f_cplx()
            loss_c.backward()
            opt_cplx.step()

    X_real_final = X_real.detach()
    X_cplx_final = X_cplx.detach()

    # 1) Final solutions match up to numerical noise: compare real vs complex.real
    assert torch.allclose(
        X_real_final,
        X_cplx_final.real,
        atol=1e-6,
        rtol=1e-6,
    )

    # 2) Loss values are (almost) identical
    final_loss_real = loss_quadratic(X_real_final, U_real).item()
    final_loss_cplx = loss_quadratic(X_cplx_final, U_cplx).item()
    assert abs(final_loss_real - final_loss_cplx) < 1e-8

    # 3) Orthonormality check for both
    Gram_real = X_real_final.T @ X_real_final
    Gram_cplx = X_cplx_final.mH @ X_cplx_final

    I_real = torch.eye(n, dtype=torch.float64, device=device)
    I_cplx = torch.eye(n, dtype=torch.complex128, device=device)

    assert torch.allclose(Gram_real, I_real, atol=1e-6, rtol=1e-6)
    assert torch.allclose(Gram_cplx, I_cplx, atol=1e-6, rtol=1e-6)
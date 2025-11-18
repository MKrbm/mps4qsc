import torch
from typing import List
from enum import Enum

def embed_iso(V: torch.Tensor, *, atol: float = 1e-7, rtol: float = 1e-5) -> torch.Tensor:
    """
    Embed a row-isometry V (M x N, with M <= N and V V^† = I_M) into a unitary U (N x N)
    such that the first M rows of U equal V. Consequently, for any |psi> ∈ C^N,

        (select-top-M-rows) · U · |psi> = V |psi>,

    which matches the block-encoding style condition ⟨0|⟨*| U |psi⟩ = V|psi⟩.

    Parameters
    ----------
    V : torch.Tensor
        Shape (M, N), real or complex. Must satisfy V @ V^† ≈ I_M.
    atol : float
        Absolute tolerance for the isometry check.
    rtol : float
        Relative tolerance for the isometry check.

    Returns
    -------
    U : torch.Tensor
        Shape (N, N) unitary with first M rows exactly equal to V.

    Raises
    ------
    ValueError
        If V is not 2D, if M > N, or if V fails the isometry check.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got ndim={V.ndim}.")
    M, N = V.shape
    if M > N:
        raise ValueError(f"Row-isometry requires M <= N, got M={M}, N={N}.")

    # Check row-isometry: V V^† ≈ I_M
    I_M = torch.eye(M, dtype=V.dtype, device=V.device)
    gram_rows = V @ (V.mH if hasattr(V, "mH") else V.conj().t())
    if not torch.allclose(gram_rows, I_M, atol=atol, rtol=rtol):
        err = torch.linalg.norm(gram_rows - I_M, ord='fro').item()
        raise ValueError(
            f"V is not a row-isometry within tolerance: ||V V^† - I||_F = {err:.3e} "
            f"(atol={atol}, rtol={rtol})."
        )

    # Complete columns of W = V^† to an ONB of C^N, then reflect back to rows of U.
    W = (V.mH if hasattr(V, "mH") else V.conj().t())  # (N, M), column-isometry
    Q_full, _ = torch.linalg.qr(W, mode="complete")   # Q_full: (N, N)
    Q_perp = Q_full[:, M:]                            # (N, N-M), orthonormal complement to span(W)

    # Columns [W, Q_perp] form a unitary; take its conjugate-transpose to get U with top M rows = V.
    U = torch.cat([W, Q_perp], dim=1)                 # (N, N) with first M columns = W
    U = U.mH if hasattr(U, "mH") else U.conj().t()    # (N, N), first M rows = V

    # (Optional) quick sanity check (should pass given the isometry check)
    I_N = torch.eye(N, dtype=U.dtype, device=U.device)
    assert torch.allclose(U.mH @ U, I_N, atol=10*atol, rtol=10*rtol)

    return U.contiguous()


def embed_non_unitary(A: torch.Tensor, *, atol: float = 1e-7, rtol: float = 1e-5) -> torch.Tensor:
    """
    Embed a square contraction A (N x N, with ||A||_2 <= 1) into a unitary U (2N x 2N)
    such that A appears as the top-left N x N block of U:

        U = [[ A,                  sqrt(I - A A^†) ],
             [ sqrt(I - A^† A),    -A^†            ]].

    This is the standard Halmos/Sz.-Nagy unitary dilation and yields a block-encoding
    in the sense that

        (⟨0| ⊗ I_N) · U · (|0⟩ ⊗ I_N) = A.

    Parameters
    ----------
    A : torch.Tensor
        Shape (N, N), real or complex. Must satisfy ||A||_2 <= 1 (within tolerance).
    atol : float
        Absolute tolerance used in internal checks and square-root regularization.
    rtol : float
        Relative tolerance used in internal checks.

    Returns
    -------
    U : torch.Tensor
        Shape (2N, 2N) unitary with top-left N x N block exactly equal to A.

    Raises
    ------
    ValueError
        If A is not 2D, not square, or not a contraction (||A||_2 > 1 + rtol).
    """

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={A.ndim}.")
    N, M = A.shape
    if N != M:
        raise ValueError(f"A must be square, got shape {A.shape}.")

    # Helper: conjugate transpose that works for real/complex and older torch versions.
    def _adjoint(X: torch.Tensor) -> torch.Tensor:
        return X.mH if hasattr(X, "mH") else X.conj().t()

    AH = _adjoint(A)

    # Check that A is a contraction: ||A||_2 <= 1 (within tolerance).
    # Use SVD to get the spectral norm (largest singular value).
    _, s, _ = torch.linalg.svd(A, full_matrices=False)
    spectral_norm = s.max().item()
    if spectral_norm > 1.0 + rtol:
        raise ValueError(
            f"A must be a contraction (||A||_2 <= 1) to admit this 2N×2N dilation, "
            f"but ||A||_2 ≈ {spectral_norm:.6f} (rtol={rtol}). "
            f"Consider rescaling A / {spectral_norm:.6f} and tracking the scale."
        )

    I_N = torch.eye(N, dtype=A.dtype, device=A.device)

    # Matrices that should be positive semidefinite:
    #   X1 = I - A A^†   (for the top-right block),
    #   X2 = I - A^† A   (for the bottom-left block).
    X1 = I_N - A @ AH
    X2 = I_N - AH @ A

    def _sqrt_psd(X: torch.Tensor) -> torch.Tensor:
        """
        Matrix square root of a Hermitian PSD matrix X using eigendecomposition.
        Small negative eigenvalues (due to numerical error) are clamped to zero.
        """
        # Symmetrize to counteract numerical non-Hermiticity.
        X_herm = 0.5 * (X + _adjoint(X))

        evals, evecs = torch.linalg.eigh(X_herm)  # evals are real
        max_abs = evals.abs().max().item() if evals.numel() > 0 else 0.0

        # If we see clearly negative eigenvalues beyond tolerance, complain.
        min_eval = evals.min().item() if evals.numel() > 0 else 0.0
        if min_eval < -max(atol, rtol * max_abs):
            raise ValueError(
                "Matrix inside the square root is not positive semidefinite "
                f"within tolerance: min eigenvalue ≈ {min_eval:.3e}."
            )

        evals_clamped = torch.clamp(evals, min=0.0)
        sqrt_evals = torch.sqrt(evals_clamped)
        sqrt_diag = torch.diag(sqrt_evals).to(X.dtype)

        sqrt_X = evecs @ sqrt_diag @ _adjoint(evecs)
        # Re-symmetrize for good measure.
        sqrt_X = 0.5 * (sqrt_X + _adjoint(sqrt_X))
        return sqrt_X

    B = _sqrt_psd(X1)  # sqrt(I - A A^†)
    C = _sqrt_psd(X2)  # sqrt(I - A^† A)

    # Assemble the 2N x 2N unitary
    top = torch.cat([A, B], dim=1)      # (N, 2N)
    bottom = torch.cat([C, -AH], dim=1) # (N, 2N)
    U = torch.cat([top, bottom], dim=0) # (2N, 2N)

    # (Optional) quick sanity check
    I_2N = torch.eye(2 * N, dtype=U.dtype, device=U.device)
    assert torch.allclose(_adjoint(U) @ U, I_2N, atol=10 * atol, rtol=10 * rtol), \
        "Constructed U is not unitary within tolerance."

    return U.contiguous()

def construct_unitary_from_As(As: List[torch.Tensor], merge_first_two: bool = True) -> tuple[List[torch.Tensor], torch.Tensor]:

    Vs = [A.reshape(-1, A.shape[-1]).T for A in As]
    AL = As[-1].reshape(-1, As[-1].shape[-1])   

    Us = [embed_iso(V) for V in Vs[:-1]]

    if merge_first_two:
        d = Us[0].shape[0]
        u0 = torch.kron(Us[0].contiguous(), torch.eye(d, dtype = Us[0].dtype))
        Us[1] = Us[1] @ u0
        Us = Us[1:]

    # SVD of last core
    U, S, V = torch.linalg.svd(AL, full_matrices=False)
    AL = U
    Us.append(embed_iso(U.T))
    last = torch.einsum("i, ij -> ij", S, V).T

    # rescale last by the largest singular value
    last = last / torch.linalg.norm(last, ord=2)
    return Us, embed_non_unitary(last)

def to_probs(outputs):
    """
    Convert outputs into probabilities (normalize along the last dimension).
    """
    return outputs / outputs.sum(dim=-1, keepdim=True)

class ManifoldType(Enum):
    CANONICAL = "canonical"
    FROBENIUS = "frobenius"
    EXACT = "exact"
    SPHERICAL = "spherical"
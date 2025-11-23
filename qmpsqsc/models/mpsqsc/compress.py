from __future__ import annotations
from typing import List, Sequence
import torch
import opt_einsum as oe  # type: ignore[import-untyped]
from tenpy.algorithms.mps_common import VariationalCompression  # type: ignore[import-untyped]
from .mpstate import MPState
from .tenpy import to_tenpy_mps, tenpy_to_mpstate



def variational_compress(mpstate: MPState, target_chi: int, max_sweeps: int = 100, tol_theta_diff: float = 1e-8) -> tuple[MPState, float]:

    """
    This is a wrapper function for TenPy's VariationalCompression.
    """


    psi = to_tenpy_mps(mpstate)
    psi_t = psi.copy()
    options = {
        "trunc_params": {
            "chi_max": int(target_chi),
        },
        "max_trunc_err" : 1.0,
        "max_sweeps": int(max_sweeps),
        "min_sweeps": 1,
        "tol_theta_diff": float(tol_theta_diff),
    }
    engine = VariationalCompression(psi_t, options)
    _ = engine.run()  # modifies `psi` in place to its best χ≤target_chi approximation
    psi_t.norm = 1

    res_mps = tenpy_to_mpstate(psi_t)
    res_mps.normalize(inplace=True)

    return res_mps, float(psi_t.overlap(psi))


def adam_refine(
    psi: MPState,   # original MPState
    phi: MPState,   # initial compressed MPState (e.g. from truncate or ALS)
    steps: int = 200,
    lr: float = 1e-3,
    normalize_each_step: bool = True,
    verbose: bool = False,
):
    """
    Gradient-based refinement of a truncated MPState using Adam.

    Maximizes the fidelity F = |<psi|phi>|^2 / (||psi||^2 ||phi||^2).

    (We enforce ||phi|| ≈ 1 by rescaling, so effectively we maximize |<psi|phi>|.
    """
    assert psi.L == phi.L
    assert psi.d == phi.d

    # Work on a detached copy
    phi_opt = phi._clone_with_As(phi.As)

    # Treat tensors as optimization parameters
    phi_opt.set_requires_grad(True)
    optimizer = torch.optim.Adam(phi_opt.As, lr=lr)

    with torch.no_grad():
        psi_norm2 = torch.real(psi.inner_product(psi))
        psi_norm = torch.sqrt(psi_norm2 + 1e-12)

    for step in range(steps):
        optimizer.zero_grad()

        ov = psi.inner_product(phi_opt)   # complex scalar ~ <phi|psi>
        # fidelity proxy: (|<psi|phi>| / ||psi||)^2, with ||phi||~1
        fid = (ov.abs() / psi_norm) ** 2
        loss = -fid   # maximize fidelity

        loss.backward()
        optimizer.step()

        if normalize_each_step:
            # Renormalize phi by rescaling a single tensor (keeps state in check)
            with torch.no_grad():
                psi.normalize(inplace=True)

        if verbose and (step + 1) % max(1, steps // 10) == 0:
            print(
                f"[adam_refine] step {step+1:4d}/{steps:4d}  "
                f"loss={loss.item():.4e}  fid={fid.item():.6f}"
            )

    phi_opt.set_requires_grad(False)
    return phi_opt

def compress_mpstate(
    psi: MPState,
    target_chi: int,
    n_sweeps: int = 100,
    adam_steps: int = 100,
    adam_lr: float = 1e-3,
) -> tuple[MPState, float]:
    """
    High-level: truncate + ALS environment sweeps + optional Adam refine.
    """
    psi.normalize(inplace=True)
    phi, fid = variational_compress(psi, target_chi, max_sweeps=n_sweeps)
    if adam_steps > 0:
        phi = adam_refine(psi, phi, steps=adam_steps, lr=adam_lr)
    
    phi.normalize(inplace=True)
    return phi, phi.overlap(psi).item()
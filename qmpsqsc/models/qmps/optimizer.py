import torch
from torch.optim.optimizer import Optimizer


inner_prod_param_dict = {'Canonical': 0.5, 'Euclidean': 0.0}

def matrix_square_root(mat_a, mat_a_size=None, iter_count=100, ridge_epsilon=1e-4):
    """
    Stable iterations for the matrix square root, Nicholas J. Higham
    Page 231, Eq 2.6b

    Works for real symmetric or complex Hermitian positive definite matrices.
    """

    if mat_a_size is None:
        mat_a_size = mat_a.shape[-1]

    identity = torch.eye(mat_a_size, device=mat_a.device, dtype=mat_a.dtype)

    # Make sure it's numerically positive definite
    mat_a = mat_a + ridge_epsilon * identity

    # 2-norm is real and >= 0 even for complex
    norm = torch.norm(mat_a, p=2)

    mat_init_y = mat_a / norm
    mat_init_z = identity
    init_err = norm

    def _iter_body(i, mat_y, unused_old_mat_y, mat_z, unused_old_mat_z, err,
                   unused_old_err):
        current_iterate = 0.5 * (3.0 * identity - torch.matmul(mat_z, mat_y))
        current_mat_y = torch.matmul(mat_y, current_iterate)
        current_mat_z = torch.matmul(current_iterate, mat_z)
        # Compute the error in approximation.
        mat_sqrt_a = current_mat_y * torch.sqrt(norm)
        mat_a_approx = torch.matmul(mat_sqrt_a, mat_sqrt_a)
        residual = mat_a - mat_a_approx
        current_err = torch.norm(residual, p=2) / norm
        return i + 1, current_mat_y, mat_y, current_mat_z, mat_z, current_err, err

    func_input = [0, mat_init_y, mat_init_y, mat_init_z, mat_init_z, init_err, init_err + 1.0]
    for _ in range(iter_count):
        func_input = _iter_body(*func_input)

    # mat_y ≈ A^{1/2} / √‖A‖, mat_z ≈ A^{-1/2} / √‖A‖
    A_root = func_input[2] * torch.sqrt(norm)
    A_root_inv = func_input[4] / torch.sqrt(norm)
    return A_root, A_root_inv


def matrix_root(A, iter_count=100):
    A_root, _ = matrix_square_root(A, A.shape[-1], ridge_epsilon=0, iter_count=iter_count)
    return A_root


def matrix_root_inv(A, iter_count=100):
    _, A_root_inv = matrix_square_root(A, A.shape[-1], ridge_epsilon=0, iter_count=iter_count)
    return A_root_inv


def cayley(Y, alpha=1.0):
    I = torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype)
    # (I - α/2 Y)^{-1} (I + α/2 Y)
    return torch.linalg.inv(I - (alpha/2.0) * Y) @ (I + (alpha/2.0) * Y)

def _update_func_Stiefel_Adam(
    X, Y, V, X_grad, p_Y, p_V, step, square, a, b,
    lr, beta_1, beta_2, expm_method, inner_iter, epsilon
):
    bias_correction_1 = 1 - beta_1 ** step
    bias_correction_2 = 1 - beta_2 ** step

    # Gram-type object with adjoint: X^* X_grad
    Xt_Xgrad = X.mH @ X_grad

    # Skew-Hermitian gradient in the Lie algebra (so exp/cayley land in unitary group)
    grad_Y = (1 - b) / 2 * (Xt_Xgrad - Xt_Xgrad.mH)

    if not square:
        # grad_V lives in the normal space, use X X^* projection with adjoint
        grad_V = -(X @ Xt_Xgrad - X_grad)

    # ----- Dynamics φ_2 (skipped when n = m) -----
    if not square:
        p_V.mul_(beta_2).add_(grad_V**2, alpha=1 - beta_2)

    # ----- Dynamics φ_1 -----
    Y.mul_(beta_1).add_(grad_Y, alpha=-(1 - beta_1))
    p_Y.mul_(beta_2).add_(grad_Y**2, alpha=1 - beta_2)

    denominator_Y = torch.sqrt(p_Y / bias_correction_2) + epsilon
    xi = lr / bias_correction_1 * Y / denominator_Y  # xi is skew-Hermitian

    if expm_method == 'Cayley':
        X.copy_(X @ cayley(xi))
    elif expm_method == 'MatrixExp':
        X.copy_(X @ torch.matrix_exp(xi))
    elif expm_method == 'ForwardEuler':
        X.add_(X @ xi)  # small step in tangent direction
    else:
        raise NotImplementedError()

    # ----- Dynamics φ_3 (skipped when n = m) -----
    if not square:
        V.mul_(beta_1)
        if beta_1 != 0:
            V.add_(V @ Y, alpha=-(3 * a - 2) / 2 * lr / beta_1)
        V.add_(grad_V, alpha=-(1 - beta_1))

        denominator_V = torch.sqrt(p_V / bias_correction_2) + epsilon
        V_tilde = V / denominator_V - X @ torch.linalg.inv(X.mH @ X) @ (X.mH @ (V / denominator_V))

        XVTV = X @ (V_tilde.mH @ V)
        X.add_(V_tilde @ (X.mH @ X), alpha=lr)
        V.add_(XVTV, alpha=-lr)

    # Final re-projection via inverse square root of X^* X (polar factor)
    gram = X.mH @ X
    X.copy_(X @ matrix_root_inv(gram, iter_count=inner_iter))

class StiefelAdam(Optimizer):
    r"""Implementation of Adam on the Stiefel / unitary manifold.

    Works for real-orthogonal (O(n)) and complex-unitary (U(n)) matrices.

    Given f(X), minimize f under constraint X^* X = I (orthonormal columns).
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), epsilon=1e-5,
                 expm_method='ForwardEuler', inner_prod='Canonical', inner_iter=10):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta_1, beta_2 = betas
        if beta_1 < 0 or beta_1 >= 1 or beta_2 < 0 or beta_2 >= 1:
            raise ValueError("beta out of range")
        assert expm_method in ['MatrixExp', 'Cayley', 'ForwardEuler'], 'expm_method not correct'

        if isinstance(inner_prod, str):
            assert inner_prod in inner_prod_param_dict.keys(), 'inner_prod not correct'
            inner_prod_param = inner_prod_param_dict[inner_prod]
        else:
            inner_prod_param = float(inner_prod)
            assert inner_prod_param < 1

        # metric parameter in Definition 1 in the paper
        a = inner_prod_param
        b = a / (a - 1)

        defaults = dict(
            lr=lr, betas=betas, epsilon=epsilon,
            expm_method=expm_method, a=a, b=b, inner_iter=inner_iter
        )
        super(StiefelAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StiefelAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            epsilon = group['epsilon']
            expm_method = group['expm_method']
            a = group['a']
            b = group['b']
            inner_iter = group['inner_iter']

            for X_raw in group['params']:
                if X_raw.grad is None:
                    continue

                # Flatten all but last two dims: shape [B, n, m]
                X = X_raw.view(-1, X_raw.shape[-2], X_raw.shape[-1])
                X_grad = X_raw.grad.view_as(X)

                # X should be tall (n >= m); otherwise transpose
                square = False
                if X.shape[-2] < X.shape[-1]:
                    X = X.transpose(-1, -2).conj()
                    X_grad = X_grad.transpose(-1, -2).conj()
                else:
                    if X.shape[-2] == X.shape[-1]:
                        square = True

                param_state = self.state[X_raw]

                if 'Y_buffer' not in param_state:
                    Y = param_state['Y_buffer'] = torch.zeros(
                        X.shape[0], X.shape[-1], X.shape[-1],
                        device=X.device, dtype=X.dtype
                    )
                if 'V_buffer' not in param_state and not square:
                    V = param_state['V_buffer'] = torch.zeros(
                        X.shape[0], X.shape[-2], X.shape[-1],
                        device=X.device, dtype=X.dtype
                    )
                if 'p_Y_buffer' not in param_state:
                    p_Y = param_state['p_Y_buffer'] = torch.zeros(
                        X.shape[0], X.shape[-1], X.shape[-1],
                        device=X.device, dtype=X.dtype
                    )
                if 'p_V_buffer' not in param_state and not square:
                    p_V = param_state['p_V_buffer'] = torch.zeros(
                        X.shape[0], X.shape[-2], X.shape[-1],
                        device=X.device, dtype=X.dtype
                    )
                if 'step' not in param_state:
                    param_state['step'] = 0

                param_state['step'] += 1
                step = param_state['step']

                Y = param_state['Y_buffer']
                p_Y = param_state['p_Y_buffer']
                if not square:
                    V = param_state['V_buffer']
                    p_V = param_state['p_V_buffer']

                if square:
                    update_func = lambda Xi, Yi, Xg, pYi: _update_func_Stiefel_Adam(
                        Xi, Yi, None, Xg, pYi, None,
                        step, square, a, b, lr, beta_1, beta_2,
                        expm_method, inner_iter, epsilon
                    )
                    torch.vmap(update_func, out_dims=None)(X, Y, X_grad, p_Y)
                else:
                    update_func = lambda Xi, Yi, Vi, Xg, pYi, pVi: _update_func_Stiefel_Adam(
                        Xi, Yi, Vi, Xg, pYi, pVi,
                        step, square, a, b, lr, beta_1, beta_2,
                        expm_method, inner_iter, epsilon
                    )
                    torch.vmap(update_func, out_dims=None)(X, Y, V, X_grad, p_Y, p_V)

                # Reshape back to original
                X_raw.copy_(X.view_as(X_raw))

        return loss

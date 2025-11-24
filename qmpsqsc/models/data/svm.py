from collections import deque
import logging
import math
from typing import Any, Callable, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..mpsqsc.mpstate import MPState
from ..mpsqsc.mpsqsc import MpsQsc
from ..qmps.qmps import qMPS
from .loss import mps_binary_predict

logger = logging.getLogger(__name__)

FeatureMap = Callable[[torch.Tensor], torch.Tensor]
BatchIterator = Iterator[Tuple[List[MPState], torch.Tensor, Any]]


class LinearSVM(nn.Module):
    """
    A simple linear SVM implemented as a single linear layer.

    The model is trained with a hinge loss plus an L2 penalty on the weights.
    """

    def __init__(self, in_dim: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, 1, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.linear(x).squeeze(-1)
        return scores

    def clone(self) -> "LinearSVM":
        """
        Create a detached copy of the current SVM (including learned weights).
        """
        device = next(self.parameters()).device
        dtype = self.linear.weight.dtype
        copy = LinearSVM(self.in_dim, dtype=dtype).to(device)
        copy.load_state_dict(self.state_dict())
        return copy


def svm_hinge_loss(model: LinearSVM,
                   scores: torch.Tensor,
                   labels: torch.Tensor,
                   C: float = 1.0) -> torch.Tensor:
    """
    Binary hinge loss with an L2 regularizer on the linear weights.
    """
    labels_pm1 = 2 * labels.float() - 1.0
    margins = 1 - labels_pm1 * scores
    hinge = torch.clamp(margins, min=0.0).mean()
    l2_reg = 0.001 * torch.sum(model.linear.weight * model.linear.weight)
    return l2_reg + C * hinge


def svm_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute 0/1 accuracy given raw SVM scores and binary labels in {0, 1}.
    """
    preds = (torch.sign(scores) > 0).long()
    return (preds == labels).float().mean()


def poly2_features(V: torch.Tensor) -> torch.Tensor:
    """
    Degree-2 homogeneous polynomial feature map for 2-way probabilities.

    Args:
        V: Tensor of shape (B, 2) containing probabilities/projections.

    Returns:
        Tensor of shape (B, 5) with features
        [x1, x2, x1^2, sqrt(2) * x1 * x2, x2^2].
    """
    x1 = V[:, 0:1]
    x2 = V[:, 1:2]
    return torch.cat(
        [x1, x2, x1 ** 2, math.sqrt(2.0) * x1 * x2, x2 ** 2],
        dim=1,
    )


def _svm_inner_loop(
    phi: torch.Tensor,
    labels: torch.Tensor,
    svm: LinearSVM,
    svm_optimizer: Optimizer,
    svm_steps: int,
    hinge_c: float,
) -> None:
    detached_phi = phi.detach()
    for _ in range(svm_steps):
        svm_optimizer.zero_grad(set_to_none=True)
        scores = svm(detached_phi)
        loss = svm_hinge_loss(svm, scores, labels, C=hinge_c)
        loss.backward()
        svm_optimizer.step()


def _hybrid_step(
    phi: torch.Tensor,
    labels: torch.Tensor,
    svm: LinearSVM,
    model_optimizer: Optimizer,
    svm_optimizer: Optimizer,
    svm_steps: int,
    hinge_c: float,
) -> Tuple[float, float]:
    _svm_inner_loop(phi, labels, svm, svm_optimizer, svm_steps, hinge_c)

    model_optimizer.zero_grad(set_to_none=True)
    scores = svm(phi)
    loss = svm_hinge_loss(svm, scores, labels, C=hinge_c)
    loss.backward()
    model_optimizer.step()
    svm_optimizer.zero_grad(set_to_none=True)

    acc = svm_accuracy(scores, labels)
    return float(loss.item()), float(acc.item())


def _run_training_loop(
    data_iterator: BatchIterator,
    compute_probs: Callable[[List[MPState]], torch.Tensor],
    svm: LinearSVM,
    model_optimizer: Optimizer,
    svm_optimizer: Optimizer,
    num_steps: int,
    svm_steps: int,
    hinge_c: float,
    feature_map: FeatureMap,
    target_loss: Optional[float] = None,
    loss_window: int = 20,
    no_improve_patience: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    track_window = (target_loss is not None) or (no_improve_patience is not None)

    if track_window and loss_window <= 0:
        raise ValueError(
            "loss_window must be positive when target_loss or patience-based stopping is used."
        )
    if no_improve_patience is not None and no_improve_patience <= 0:
        raise ValueError("no_improve_patience must be positive when provided.")

    losses: List[float] = []
    accuracies: List[float] = []
    window: Optional[deque] = deque() if track_window else None
    window_sum = 0.0
    best_window_avg = math.inf
    steps_since_best = 0

    logger.debug(
        "Starting hybrid training loop: steps=%d, svm_steps=%d, target_loss=%s",
        num_steps,
        svm_steps,
        target_loss,
    )

    for step in range(num_steps):
        states, labels, *_ = next(data_iterator)
        probs = compute_probs(states)
        phi = feature_map(probs)
        loss_value, acc_value = _hybrid_step(
            phi, labels, svm, model_optimizer, svm_optimizer, svm_steps, hinge_c
        )
        losses.append(loss_value)
        accuracies.append(acc_value)

        message = f"Hybrid step {step + 1}/{num_steps} loss={loss_value:.6f} acc={acc_value:.3f}"

        if window is not None:
            window.append(loss_value)
            window_sum += loss_value
            if len(window) > loss_window:
                window_sum -= window.popleft()
            if len(window) == loss_window:
                avg = window_sum / loss_window
                if target_loss is not None:
                    message += (
                        f" (window={loss_window} avg={avg:.6f} <= target={target_loss})"
                    )
                    if avg <= target_loss:
                        logger.info(
                            "Early stop (target) at step %d (window=%d avg=%.6f <= target=%.6f)",
                            step + 1,
                            loss_window,
                            avg,
                            target_loss,
                        )
                        break
                if no_improve_patience is not None:
                    if avg < best_window_avg - 1e-12:
                        best_window_avg = avg
                        steps_since_best = 0
                    else:
                        steps_since_best += 1
                        if steps_since_best >= no_improve_patience:
                            logger.info(
                                "Early stop (patience) at step %d (no window avg improvement "
                                "for %d steps, best=%.6f, current=%.6f)",
                                step + 1,
                                no_improve_patience,
                                best_window_avg,
                                avg,
                            )
                            break
        logger.info(message)

    return losses, accuracies


def _evaluate_qmps_loss(
    model: qMPS,
    svm: LinearSVM,
    data_iterator: BatchIterator,
    feature_map: FeatureMap,
    hinge_c: float,
    num_batches: int,
) -> float:
    if num_batches <= 0:
        return 0.0

    logger.debug(
        "Evaluating qMPS loss over %d batches (hinge_c=%s)",
        num_batches,
        hinge_c,
    )

    total = 0.0
    count = 0
    prev_mode = getattr(model, "training", True)
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(num_batches):
                states, labels, *_ = next(data_iterator)
                probs, _ = model.predict(states)
                phi = feature_map(probs)
                scores = svm(phi)
                loss = svm_hinge_loss(svm, scores, labels, C=hinge_c)
                total += float(loss.item())
                count += 1
    finally:
        model.train(prev_mode)
    return total / max(1, count)


def _select_weight_for_qmps(
    model: qMPS,
    svm: LinearSVM,
    data_iterator: BatchIterator,
    feature_map: FeatureMap,
    hinge_c: float,
    max_weight: float,
    num_eval_batches: int,
    loss_threshold: float | None,
    initial_weight: float,
    search_tol: float,
    max_iterations: int,
) -> Tuple[float, List[Tuple[float, float]]]:
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    upper_bound = min(max_weight, 1.0)
    initial_weight = float(initial_weight)
    if initial_weight < 0:
        initial_weight = 0.0
    if initial_weight >= upper_bound:
        initial_weight = upper_bound - 1e-6

    eval_log: List[Tuple[float, float]] = []

    def eval_weight(weight: float) -> float:
        clamped = min(max(weight, 0.0), upper_bound - 1e-6)
        model.set_weights(clamped)
        avg_loss = _evaluate_qmps_loss(
            model,
            svm,
            data_iterator,
            feature_map,
            hinge_c,
            num_eval_batches,
        )
        eval_log.append((clamped, avg_loss))
        return avg_loss

    if loss_threshold is None:
        selected = upper_bound - 1e-6
        eval_weight(selected)
        model.set_weights(selected)
        return selected, eval_log

    lo_w = initial_weight
    lo_loss = eval_weight(lo_w)
    if lo_loss > loss_threshold:
        lo_w = 0.0
        lo_loss = eval_weight(lo_w)
        if lo_loss > loss_threshold:
            logger.warning(
                "Minimum weight (w=0.0) still exceeds loss threshold (%.6f > %.6f)",
                lo_loss,
                loss_threshold,
            )
            model.set_weights(lo_w)
            return lo_w, eval_log

    hi_w = upper_bound - 1e-6
    hi_loss = eval_weight(hi_w)
    best_w = lo_w

    if hi_loss <= loss_threshold:
        best_w = hi_w
    else:
        left_w = lo_w
        right_w = hi_w
        for _ in range(max_iterations):
            if right_w - left_w <= search_tol:
                break
            mid_w = 0.5 * (left_w + right_w)
            mid_loss = eval_weight(mid_w)
            if mid_loss <= loss_threshold:
                left_w = mid_w
                if mid_w > best_w:
                    best_w = mid_w
            else:
                right_w = mid_w
            
            logger.debug(
                "Weight search: left=%.6f, right=%.6f, mid=%.6f, mid_loss=%.6f",
                left_w,
                right_w,
                mid_w,
                mid_loss,
            )

    logger.info(
        "Selected weight w=%.6f after %d evaluations",
        best_w,
        len(eval_log),
    )
    model.set_weights(best_w)
    return best_w, eval_log


def train_mpstates_with_svm(
    positive_state: MPState,
    negative_state: MPState,
    data_iterator: BatchIterator,
    svm: LinearSVM,
    mp_optimizer: Optimizer,
    svm_optimizer: Optimizer,
    *,
    num_steps: int,
    svm_steps: int = 30,
    hinge_c: float = 0.5,
    feature_map: FeatureMap = poly2_features,
) -> Tuple[List[float], List[float]]:
    """
    Train two MPState instances (e.g. GHZ vs product) with an auxiliary SVM.
    """

    def _compute_probs(states: List[MPState]) -> torch.Tensor:
        probs, _ = mps_binary_predict(positive_state, negative_state, states)
        return probs

    return _run_training_loop(
        data_iterator=data_iterator,
        compute_probs=_compute_probs,
        svm=svm,
        model_optimizer=mp_optimizer,
        svm_optimizer=svm_optimizer,
        num_steps=num_steps,
        svm_steps=svm_steps,
        hinge_c=hinge_c,
        feature_map=feature_map,
        no_improve_patience=None,
    )


def train_mpsqsc_with_svm(
    model: MpsQsc,
    data_iterator: BatchIterator,
    svm: LinearSVM,
    model_optimizer: Optimizer,
    svm_optimizer: Optimizer,
    *,
    num_steps: int,
    svm_steps: int = 30,
    hinge_c: float = 0.5,
    feature_map: FeatureMap = poly2_features,
) -> Tuple[List[float], List[float]]:
    """
    Train an MpsQsc model jointly with an SVM using probability features.
    """

    def _compute_probs(states: List[MPState]) -> torch.Tensor:
        probs, _ = model.predict(states)
        return probs

    return _run_training_loop(
        data_iterator=data_iterator,
        compute_probs=_compute_probs,
        svm=svm,
        model_optimizer=model_optimizer,
        svm_optimizer=svm_optimizer,
        num_steps=num_steps,
        svm_steps=svm_steps,
        hinge_c=hinge_c,
        feature_map=feature_map,
        no_improve_patience=None,
    )


def train_qmps_with_svm(
    model: qMPS,
    data_iterator: BatchIterator,
    svm: LinearSVM,
    model_optimizer: Optimizer,
    svm_optimizer: Optimizer,
    *,
    num_steps: int,
    svm_steps: int = 10,
    hinge_c: float = 0.5,
    feature_map: FeatureMap = poly2_features,
    max_weight: float = 1.0,
    weight_eval_batches: int = 5,
    weight_search_tol: float = 5e-4,
    weight_search_steps: int = 15,
    max_weight_loops: int = 8,
    initial_weight: Optional[float] = None,
    weight_loss_threshold: float = 0.15,
    target_loss: float = 0.15,
    loss_window: int = 50,
    no_improve_patience: Optional[int] = 20,
) -> Tuple[List[float], List[float], dict[str, Any]]:
    """
    Train a qMPS model using the shared hybrid SVM loop.

    Additionally supports searching over classifier weights `w` before training
    and stopping early once a moving-average loss threshold is achieved or when
    the moving-average loss fails to improve for `no_improve_patience` steps.
    """

    def _compute_probs(states: List[MPState]) -> torch.Tensor:
        probs, _ = model.predict(states)
        return probs

    metadata: dict[str, Any] = {}
    start_weight = (
        float(initial_weight)
        if initial_weight is not None
        else getattr(model, "current_weight", 0.0)
    )
    prior_weight = start_weight
    loss_threshold = weight_loss_threshold

    all_losses: List[List[float]] = []
    all_accs: List[List[float]] = []
    accumulated_losses: List[float] = []
    accumulated_accs: List[float] = []
    weight_history: List[float] = []
    eval_history: List[List[Tuple[float, float]]] = []
    steps_per_loop: List[int] = []

    logger.info(
        "Starting qMPS hybrid training with weight target=%.3f (initial=%.3f)",
        max_weight,
        prior_weight,
    )

    for loop_idx in range(max_weight_loops):
        if loop_idx == 0 and start_weight != 0:
            selected_weight = prior_weight
            eval_log = [(0.0, 0.0)]
        else:
            selected_weight, eval_log = _select_weight_for_qmps(
                model=model,
                svm=svm,
                data_iterator=data_iterator,
                feature_map=feature_map,
                hinge_c=hinge_c,
                max_weight=max_weight,
                num_eval_batches=weight_eval_batches,
                loss_threshold=loss_threshold,
                initial_weight=prior_weight,
                search_tol=weight_search_tol,
                max_iterations=weight_search_steps,
            )
        weight_history.append(selected_weight)
        eval_history.append(eval_log)

        losses_iter, accs_iter = _run_training_loop(
            data_iterator=data_iterator,
            compute_probs=_compute_probs,
            svm=svm,
            model_optimizer=model_optimizer,
            svm_optimizer=svm_optimizer,
            num_steps=num_steps,
            svm_steps=svm_steps,
            hinge_c=hinge_c,
            feature_map=feature_map,
            target_loss=target_loss,
            loss_window=loss_window,
            no_improve_patience=no_improve_patience,
        )
        all_losses.append(losses_iter)
        all_accs.append(accs_iter)
        accumulated_losses.extend(losses_iter)
        accumulated_accs.extend(accs_iter)
        steps_per_loop.append(len(losses_iter))

        logger.info(
            "Completed weight loop %d/%d (w=%.6f, steps=%d)",
            loop_idx + 1,
            max_weight_loops,
            selected_weight,
            len(losses_iter),
        )

        if selected_weight >= max_weight - weight_search_tol:
            logger.info("Reached desired weight %.3f; stopping outer loop.", selected_weight)
            break

        prior_weight = selected_weight
    else:
        logger.warning(
            "Max weight loops (%d) reached without hitting target weight %.3f",
            max_weight_loops,
            max_weight,
        )

    metadata["initial_weight"] = start_weight
    metadata["selected_weight"] = weight_history[-1] if weight_history else metadata["initial_weight"]
    metadata["all_selected_weights"] = weight_history
    metadata["weight_evaluations"] = eval_history[-1] if eval_history else []
    metadata["weight_evaluations_history"] = eval_history
    metadata["losses_per_loop"] = all_losses
    metadata["accs_per_loop"] = all_accs
    metadata["steps_per_loop"] = steps_per_loop
    metadata["loops_completed"] = len(weight_history)

    metadata["steps"] = len(accumulated_losses)
    metadata["final_window_loss"] = (
        sum(accumulated_losses[-loss_window:]) / loss_window
        if len(accumulated_losses) >= loss_window
        else None
    )

    return accumulated_losses, accumulated_accs, metadata


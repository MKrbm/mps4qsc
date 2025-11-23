import torch
from ..qmps.qmps import qMPS
from ..mpsqsc.mpstate import MPState
from ..mpsqsc.mpsqsc import MpsQsc
from typing import List, Tuple

def mps_binary_predict(mps1: MPState, mps2: MPState, states: List[MPState]) -> Tuple[torch.Tensor, torch.Tensor]:
    # The best way is to create a list of Tensor pairs and stack them along a new dimension
    amps = torch.stack([
        torch.stack([mps1.overlap(s), mps2.overlap(s)])
        for s in states
    ], dim=0)
    amps += 1e-12
    norms = amps.norm(dim=-1)
    amps = amps / norms[:, None]
    probs = amps*amps.conj()
    return probs.real, norms.real


def calculate_loss_mpstates(mps1: MPState, mps2: MPState, states: List[MPState], labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    preds, norms = mps_binary_predict(mps1, mps2, states)
    probs = preds[torch.arange(len(labels)), labels]
    acc = (probs > 0.5).float()
    loss = -torch.log(probs).mean()
    return loss, acc

def calculate_loss_mpsqsc(mpsqsc: MpsQsc, states: List[MPState], labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    preds, norms = mpsqsc.predict(states)
    probs = preds[torch.arange(len(labels)), labels]
    acc = (probs > 0.5).float()
    loss = -torch.log(probs).mean()
    return loss, acc

def calculate_loss_qmps(qmps: qMPS, states: List[MPState], labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs, norms = qmps.predict(states)
    probs = probs[torch.arange(len(labels)), labels]
    acc = (probs > 0.5).float()
    loss = -torch.log(probs).mean()
    return loss, acc, probs
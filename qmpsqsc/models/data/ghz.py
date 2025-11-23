import torch
from typing import List, Tuple, Iterator
from ..mpsqsc.mpstate import MPState

@torch.no_grad()
def create_ghz_rho_batch_qsc(
    mpsghz: MPState,
    mps_allup: MPState,
    mps_alldown: MPState,
    batch_size: int,
    error_rate: float,
) -> Iterator[Tuple[List[MPState], torch.Tensor, List[int]]]:
    """
    Create an infinite generator of training batches for GHZ vs rho with local bit-flip errors.

    States composition:
        - 50% GHZ
        - 25% all-up
        - 25% all-down

    For each sample:
        1. Draw C ~ Poisson(L * error_rate) (L = number of sites).
        2. Choose C distinct sites uniformly at random.
        3. Flip the local tensor at those sites (Pauli-X in physical dimension).

    Yields:
        states: list of length batch_size, each an MPS-like object
        labels: LongTensor of shape (batch_size,) on `device`
                0 -> GHZ   (entangled)
                1 -> rho   (product states: all-up / all-down)
    """

    def _flip_sites_in_mps(mps_state, site_indices):
        """
        In-place flip of given sites in an MPS.

        Assumes:
            - mps_state[site] is a torch.Tensor
            - physical index is dimension 1 and has size 2
        """
        As = mps_state.As
        X = torch.tensor([[0, 1], [1, 0]], device=As[0].device, dtype=As[0].dtype)
        L = len(mps_state.As)
        for i in site_indices:
            A = mps_state.As[i]
            if i != 0 and i != L - 1:
                A.data[:] = torch.einsum("iaj, ab -> ibj", A.data, X)
            elif i == 0:
                A.data[:] = torch.einsum("aj, ab -> bj", A.data, X)
            else:  # ind == L - 1
                A.data[:] = torch.einsum("ia, ab -> ib", A.data, X)
            mps_state.As[i] = A

    device = mpsghz.device
    num_sites = mpsghz.L

    while True:
        # ---- Build the desired mixture: 50% GHZ, 25% all-up, 25% all-down ----
        num_ghz = batch_size // 2
        remaining = batch_size - num_ghz
        num_allup = remaining // 2
        num_alldown = remaining - num_allup

        # 0 = GHZ, 1 = all-up, 2 = all-down (internal codes)
        _type_codes = (
            [0] * num_ghz +
            [1] * num_allup +
            [2] * num_alldown
        )
        type_codes = torch.tensor(_type_codes, device=device)

        # Shuffle to avoid any ordering bias
        perm = torch.randperm(batch_size, device=device)
        type_codes = type_codes[perm]

        states: List = []
        labels: List[int] = []
        errors: List[int] = [] # 0: no error, 1: error

        for idx in range(batch_size):
            code = int(type_codes[idx].item())

            if code == 0:
                base_state = mpsghz
                label = 0  # GHZ
            elif code == 1:
                base_state = mps_allup
                label = 1  # product
            else:  # code == 2
                base_state = mps_alldown
                label = 1  # product

            state = base_state.copy()
            # Randomly flip a site with probability error_rate for this sample
            if torch.rand(1).item() < error_rate:
                errors.append(1)
                site = torch.randint(0, num_sites, (1,)).item()
                _flip_sites_in_mps(state, [site])
            else:
                errors.append(0)
            states.append(state)
            labels.append(label)

        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        yield states, labels_tensor, errors

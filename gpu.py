"""Small GPU helper."""

import torch


def has_bf16():
    # bf16 needs a newer GPU (A100, L4). The T4 doesn't support it, even
    # though torch.cuda.is_bf16_supported() can return True on it.
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8

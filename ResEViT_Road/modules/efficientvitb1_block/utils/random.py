import torch

__all__ = [
    "torch_random",
    "torch_uniform",
]


def torch_random(generator: torch.Generator or None = None) -> float:
    """uniform distribution on the interval [0, 1)"""
    return float(torch.rand(1, generator=generator))



def torch_uniform(low: float, high: float, generator: torch.Generator or None = None) -> float:
    """uniform distribution on the interval [low, high)"""
    rand_val = torch_random(generator)
    return (high - low) * rand_val + low
__all__ = [
    "list_sum",
    "weighted_list_sum",
    "val2list",
    "val2tuple",
]


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def weighted_list_sum(x: list, weights: list) -> any:
    assert len(x) == len(weights)
    return x[0] * weights[0] if len(x) == 1 else x[0] * weights[0] + weighted_list_sum(x[1:], weights[1:])


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

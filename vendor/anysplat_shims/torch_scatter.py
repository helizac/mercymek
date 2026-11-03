"""Pure-torch stand-ins for the two torch_scatter ops AnySplat's encoder uses (runs on XPU/CPU)."""
import torch


def _size(index, dim_size):
    return int(index.max()) + 1 if dim_size is None else dim_size


def scatter_add(src, index, dim=0, dim_size=None):
    assert dim == 0
    out = src.new_zeros((_size(index, dim_size),) + src.shape[1:])
    idx = index.long().view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_max(src, index, dim=0, dim_size=None):
    assert dim == 0
    out = src.new_full((_size(index, dim_size),) + src.shape[1:], float("-inf"))
    idx = index.long().view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True), None

"""memory_efficient_attention on torch SDPA ((B, N, H, D) layout like xformers)."""
from types import SimpleNamespace

import torch.nn.functional as F

fmha = SimpleNamespace(flash=SimpleNamespace(FwOp=None, BwOp=None))


def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, op=None, **_):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=p)
    return o.transpose(1, 2)

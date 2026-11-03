"""flash_attn_func on torch SDPA, same (batch, seqlen, heads, dim) layout as flash-attn."""
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2)

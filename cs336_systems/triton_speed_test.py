import torch
import triton

from cs336_systems.flash_attention import FlashAttention2
from typing import Callable

# problem size
n_heads = 16
d_head = 64
sequence_length = 16384

# q, k, v with grads, on GPU, BF16
q, k, v = torch.randn(
    3, n_heads, sequence_length, d_head,
    device='cuda', dtype=torch.bfloat16, requires_grad=True
).unbind(0)

# compile the custom autograd Function's .apply
flash: Callable[..., torch.Tensor]
flash = torch.compile(FlashAttention2.apply)


def flash_forward_backward():
    o = flash(q, k, v, True)  # causal=True
    loss = o.sum()
    loss.backward()

results = triton.testing.do_bench(flash_forward_backward, warmup=100, rep=500)
print(results)
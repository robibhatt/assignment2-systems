import torch
import triton
import triton.language as tl
from typing import Callable, cast
from torch import Tensor




class FlashAttention2(torch.autograd.Function):
    """
    Dummy FlashAttention-2 autograd.Function scaffold.

    Forward signature: (Q, K, V, is_causal=False) -> O
      - Q, K, V: (..., d)
      - is_causal: bool (no grad)

    Backward signature: (*grad_outputs) -> (dQ, dK, dV, None)

    This version is intentionally WRONG for math: it just returns zeros,
    but the shapes, autograd wiring, and ctx usage are correct so you can
    implement kernels later without changing call sites.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        # Save tensors/flags for backward (you'll likely also save O and L later)
        ctx.save_for_backward(Q, K, V)
        ctx.is_causal = is_causal

        # Output tensor (correct shape, wrong values on purpose)
        O = torch.zeros_like(Q)

        # In the real impl you'll also compute L (logsumexp per row) and save it:
        # ctx.L = <tensor with shape Q.shape[:-1]>

        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Unpack gradient wrt output
        (dO,) = grad_outputs

        # Retrieve saved tensors/flags
        Q, K, V = ctx.saved_tensors
        # is_causal = ctx.is_causal  # likely needed in your real backward

        # Dummy gradients matching input shapes
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Return grads for (Q, K, V, is_causal). No grad for bool flag -> None.
        return dQ, dK, dV, None


# Convenient alias: typical call pattern is FlashAttention2Fn(Q, K, V, is_causal)
FlashAttention2Fn = FlashAttention2.apply


# --- Optional: compile handle you can swap in tests/benchmarks later ---
# (You can comment this out if you prefer to compile at call site.)
flash_attention2_compiled = torch.compile(FlashAttention2.apply)


# --- Optional smoke test ---

FlashFn = Callable[[Tensor, Tensor, Tensor, bool], Tensor]
flash: FlashFn = cast(FlashFn, FlashAttention2.apply)
if __name__ == "__main__":
    q, k, v = [torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    o = flash(q, k, v, True)
    print("O shape:", o.shape)
    o.sum().backward()
    assert q.grad is not None
    print("q.grad shape:", q.grad.shape)

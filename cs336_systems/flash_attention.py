import torch
import triton
import triton.language as tl
from typing import Callable, cast
from torch import Tensor




class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:

        
        n_heads, seq_len, d = Q.shape
        device = Q.device
        dtype = Q.dtype

        return Q

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
    

FlashAttention2Fn = FlashAttention2.apply
flash_attention2_compiled = torch.compile(FlashAttention2.apply)

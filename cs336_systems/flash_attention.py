from ast import Pass
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

        batch_q = 16
        batch_k = 16

        q_tiles = seq_len // batch_q
        k_tiles = seq_len // batch_k

        output = torch.zeros_like(V)
        L = torch.zeros((n_heads, seq_len), dtype=dtype)
        M = torch.full((n_heads, seq_len), -torch.inf, dtype=dtype)

        for head_index in range(n_heads):
            for q_tile in range(q_tiles):
                pass
                


        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
    

FlashAttention2Fn = FlashAttention2.apply
flash_attention2_compiled = torch.compile(FlashAttention2.apply)

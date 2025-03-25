import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl

DEVICE = torch.cuda.current_device()

@triton.jit
def _topk_retr(
    q, k, topk_scores, topk_indices,
    q_stride0, q_stride1, q_stride2, q_stride3, q_stride4,
    k_stride0, k_stride1, k_stride2, k_stride3,
    topk_indices_stride0, topk_indices_stride1, topk_indices_stride2, topk_indices_stride3, 
    Z, H, GRP_SIZE, N_CTX,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) 
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    k_offset = off_z.to(tl.int64) * k_stride0 + off_h.to(tl.int64) * k_stride1  # [seq_len, head_dim]
    q_offset = off_z.to(tl.int64) * q_stride0 + off_h.to(tl.int64) * q_stride1  # [seq_len, query_grp_size, head_dim]
    topk_offset = off_z.to(tl.int64) * topk_indices_stride0 + off_h.to(tl.int64) * topk_indices_stride1  # [seq_len, topk]

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=q + q_offset,
        shape=(N_CTX, GRP_SIZE, HEAD_DIM),
        strides=(q_stride2, q_stride3, q_stride4),
        offsets=(start_m * BLOCK_M, 0, 0),
        block_shape=(BLOCK_M, GRP_SIZE, HEAD_DIM),
        order=(2, 1, 0)
    )
    # k is transposed
    K_block_ptr = tl.make_block_ptr(
        base=k + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(k_stride3, k_stride2),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    topk_scores_block_ptr = tl.make_block_ptr(
        base=topk_scores + topk_offset,
        shape=(N_CTX, TOPK),
        strides=(topk_indices_stride2, topk_indices_stride3),
        offsets=(0, 0),
        block_shape=(BLOCK_M, TOPK),
        order=(1, 0),
    )
    topk_indices_block_ptr = tl.make_block_ptr(
        base=topk_indices + topk_offset,
        shape=(N_CTX, TOPK),
        strides=(topk_indices_stride2, topk_indices_stride3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, TOPK),
        order=(1, 0),
    )
    # initialize offsets
    offs_q = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k =  tl.arange(0, BLOCK_N) 
    offs_topk = start_m * BLOCK_M + tl.arange(0, BLOCK_M)



class _topk_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, topk, sm_scale):
        """
        q: (batch_size, num_heads, seq_len, head_dim)
        k: (batch_size, num_kv_heads, seq_len, head_dim)
        v: (batch_size, num_kv_heads, seq_len, head_dim)
        """
        input_q_shape = q.shape
        BATCH, NUM_HEADS, N_CTX, HEAD_DIM = input_q_shape
        NUM_KV_HEADS = k.shape[1]
        assert NUM_KV_HEADS <= NUM_HEADS and NUM_HEADS % NUM_KV_HEADS == 0
        GRP_SIZE = NUM_HEADS // NUM_KV_HEADS
        assert N_CTX == k.shape[2] 

        q = rearrange(q, 'b (h r) s d -> b h s r d', r=GRP_SIZE).contiguous()
        topk_scores = torch.empty(*q.shape[:-1], topk, device=q.device, dtype=torch.float32)
        topk_indices = torch.empty(*k.shape[:-1], topk, device=k.device, dtype=torch.int32)
        o = torch.empty_like(q)
        M = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_SIZE"]), BATCH * NUM_KV_HEADS, 1)
        _topk_retr[grid](
            q, k, topk_scores, topk_indices,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2), topk_indices.stride(3),
            BATCH, NUM_KV_HEADS, GRP_SIZE,
            N_CTX=k.shape[2],
            HEAD_DIM=HEAD_DIM,
            TOPK=topk,
        )

        _attn_fwd[grid](
            topk_scores, v, sm_scale, M, o,
            topk_scores.stride(0), topk_scores.stride(1), topk_scores.stride(2), topk_scores.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            BATCH, NUM_KV_HEADS, GRP_SIZE,
            N_CTX=k.shape[2],
            HEAD_DIM=HEAD_DIM,
        )
        o = rearrange(o, 'b h s r d -> b (h r) s d', r=GRP_SIZE).contiguous()
        ctx.save_for_backward(q, k, v, topk_indices, o, M)
        ctx.topk = topk
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.NUM_KV_HEADS = NUM_KV_HEADS
        ctx.GRP_SIZE = GRP_SIZE
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, topk_indices, o, M = ctx.saved_tensors
        topk = ctx.topk
        sm_scale = ctx.sm_scale
        HEAD_DIM = ctx.HEAD_DIM
        NUM_KV_HEADS = ctx.NUM_KV_HEADS
        GRP_SIZE = ctx.GRP_SIZE

        o = rearrange(o, 'b (h r) s d -> b h s r d', r=GRP_SIZE).contiguous()
        do = rearrange(do, 'b (h r) s d -> b h s r d', r=GRP_SIZE).contiguous()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        BATCH, _, N_CTX, _ = k.shape
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 4
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k 
        arg_k = k
        arg_k = arg_k * (sm_scale * RCP_LN2)
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * NUM_KV_HEADS) 
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, NUM_KV_HEADS, GRP_SIZE,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM,
        )

        grid = (N_CTX // BLOCK_N1, 1, BATCH * NUM_KV_HEADS)
        _attn_bwd[grid](
            q, arg_k, topk_indices, v, do, dq, dk, dv,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        dq = rearrange(dq, 'b h s r d -> b (h r) s d', r=GRP_SIZE).contiguous()
        return dq, dk, dv, None, None
    
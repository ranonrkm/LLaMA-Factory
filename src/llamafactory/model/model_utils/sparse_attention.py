from typing import Optional, Tuple, Callable
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging


logger = logging.get_logger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def sparse_attn_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        seq_len = hidden_states.size(1)
        
        sink = self.sink    #4
        local = self.local  #1024
        topk = self.topk    #256
        topp = self.topp    #0.95
        assert not (topk > 0 and topp > 0), "topk and topp cannot be both set to a value greater than 0"
        n_full_attn = sink + local + topk
        if topp > 0:
            n_full_attn += local
        
        sliding_window = None
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        dropout = 0.0 if not self.training else self.attention_dropout
        attn_mask = None
        
        if self.layer_idx > 0 and seq_len > n_full_attn:
            num_kv_heads = self.config.num_key_value_heads
            attn_mask = torch.ones(*key_states.shape[:-1], seq_len, device=hidden_states.device).to(torch.bool)
            attn_mask.tril_(0)

            if topk > 0 or topp > 0:
                chunk_size = 1024
                num_chunks = (seq_len - n_full_attn) // chunk_size
                for i in range(num_chunks): 
                    start = n_full_attn + i * chunk_size
                    end = seq_len if i == num_chunks - 1 else start + chunk_size
                    dynamic_mask = torch.ones(*key_states.shape[:2], end - start, end, device=hidden_states.device).to(torch.bool)
                    dynamic_mask.tril_(start - local)
                    dynamic_mask[..., :sink] = False
                    attn_mask[..., start:end, :end].logical_xor_(dynamic_mask)

                    with torch.no_grad():
                        mean_query_states = rearrange(query_states[..., start:end, :], 'b (h r) n d -> b h r n d', h=num_kv_heads).mean(dim=2)
                        attn = torch.einsum('b h n d, b h m d -> b h n m', mean_query_states, key_states.narrow(2, 0, end))
                        attn.masked_fill_(dynamic_mask.logical_not(), -1e9)
                        if topk > 0:
                            topk_ids = attn.topk(topk, dim=-1).indices
                            attn_mask[..., start:end, :end].scatter_(dim=-1, index=topk_ids, value=True)
                        else:
                            attn = attn / math.sqrt(self.head_dim)
                            attn = F.softmax(attn, dim=-1, dtype=torch.float32)
                            sorted_attn, sorted_ids = attn.sort(dim=-1, descending=True)
                            sorted_attn = sorted_attn.cumsum(dim=-1)
                            select_mask = torch.zeros_like(sorted_attn, dtype=torch.bool)
                            select_mask[..., 1:] = sorted_attn[..., :-1] < topp
                            select_mask[..., 0] = True
                            attn_mask[..., start:end, :end].scatter_(dim=-1, index=sorted_ids, value=select_mask)

            else:
                attn_mask[..., n_full_attn:, sink:].triu_(1)

            num_kv_groups = self.num_key_value_groups
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1)
            attn_mask = rearrange(attn_mask, 'b h r t n -> b (h r) t n')

            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  
            # dropout = 0.0

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attn_mask,    
            dropout=dropout,    # (not self.training or self.layer_idx > 0)
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )  

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def sparse_attn_forward_approx(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        seq_len = hidden_states.size(1)
        
        sink = 4
        local = 1024
        topk = 256
        sliding_window = None
        if self.layer_idx > 0 and (seq_len - sink - local) > topk:
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  
            num_kv_heads = self.config.num_key_value_heads
            n_full_attn = sink + local + topk
            chunk_size = n_full_attn
            num_chunks = seq_len // chunk_size
            attn_outputs = []      

            for i in range(num_chunks):
                start = i * chunk_size
                end = seq_len if i == num_chunks - 1 else start + chunk_size
                query_states_chunk = query_states.narrow(2, start, end - start)
                if i == 0:
                    key_states_chunk = key_states.narrow(2, 0, end)
                    value_states_chunk = value_states.narrow(2, 0, end)
                    attn_mask = None
                else:
                    key_states_chunk = torch.cat([
                        key_states.narrow(2, 0, start).detach(),
                        key_states.narrow(2, start, end - start),
                    ], dim=2)
                    value_states_chunk = torch.cat([
                        value_states.narrow(2, 0, start).detach(),
                        value_states.narrow(2, start, end - start),
                    ], dim=2)

                    with torch.no_grad():
                        attn_mask = torch.ones(end - start, end, device=hidden_states.device).to(torch.bool)
                        attn_mask.tril_(start)
                        dynamic_attn_mask = torch.ones(end - start, end - local - sink, device=hidden_states.device).to(torch.bool)
                        dynamic_attn_mask.tril_(start - local - sink)

                        mean_query_states = rearrange(query_states_chunk, 'b (h r) n d -> b h r n d', h=num_kv_heads).mean(dim=2)
                        attn = torch.einsum('b h n d, b h m d -> b h n m', mean_query_states, key_states_chunk.narrow(2, sink, end - sink - local))
                        attn.masked_fill_(dynamic_attn_mask.logical_not(), -1e9)
                        topk_ids = attn.topk(topk, dim=-1).indices

                        attn_mask[..., sink : -local].logical_xor_(dynamic_attn_mask)
                        # attn_mask = attn_mask[None, None, ...].expand(*topk_ids.shape[:2], -1, -1)
                        attn_mask = attn_mask[None, None, ...].repeat(*topk_ids.shape[:2], 1, 1)
                        attn_mask[..., sink : -local].scatter_(dim=-1, index=topk_ids, value=True)
                        attn_mask = rearrange(attn_mask.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1), 'b h r t n -> b (h r) t n')

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states_chunk,
                    key_states_chunk,
                    value_states_chunk,
                    attention_mask=attn_mask,  # the diff with full attention
                    dropout=0.0 if not self.training else self.attention_dropout,    # (not self.training or self.layer_idx > 0)
                    scaling=self.scaling,
                    sliding_window=sliding_window,  # main diff with Llama
                    **kwargs,
                )

                attn_outputs.append(attn_output)

            attn_output = torch.cat(attn_outputs, dim=1)    # the orientation is changed from B H N D to B N H D in transformers 4.48

        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,    
                dropout=0.0 if not self.training else self.attention_dropout,    # (not self.training or self.layer_idx > 0)
                scaling=self.scaling,
                sliding_window=sliding_window,  # main diff with Llama
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


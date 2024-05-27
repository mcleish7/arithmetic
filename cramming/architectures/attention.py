"""Attention modules. Most code heavily stolen from the GPT-neoX implementation"""
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA, FIRE
from typing import Optional

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # use to mark output projections of attn while it exists


def get_attention_mechanism(idx, hidden_size, cfg_attention, norm_fn: torch.nn.Identity):
    # ########## main implementation
    if cfg_attention.type == "self-attention":
        mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention, norm_fn)  # neox
    # ########## other things:
    elif cfg_attention.type == "pytorch":
        mechanism = SelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "pytorch-seqfirst":
        mechanism = SeqFirstSelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "huggingface":
        mechanism = BertAttentionWrapper(hidden_size, cfg_attention)  # always includes bias!
    elif cfg_attention.type == "fourier":
        mechanism = FourierMixing(hidden_size, cfg_attention)
    elif cfg_attention.type == "none":
        mechanism = Identity(hidden_size)
    elif cfg_attention.type == "rn":
        mechanism = RandomNoise(hidden_size) # i.e. no signal on where to look
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism


class Identity(torch.nn.Module):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size):
        super().__init__()
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return hidden_states

class RandomNoise(torch.nn.Module):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size):
        super().__init__()
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        print("using rn")
        return hidden_states + torch.normal(0, 0.1, hidden_states.shape).to(hidden_states.device)

class BertAttentionWrapper(BertSelfAttention):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        class config:
            pass

        config.hidden_size = hidden_size
        config.num_attention_heads = cfg_attention.num_attention_heads
        config.attention_probs_dropout_prob = 0.0
        config.is_decoder = True

        super().__init__(config)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(hidden_size, hidden_size, bias=cfg_attention.bias_in_proj)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(super().forward(hidden_states, attention_mask)[0])


class SelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=0.0,
            batch_first=True,
            bias=cfg_attention.bias_in_proj,
            add_bias_kv=cfg_attention.qkv_bias,
        )

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask[0, 0, :, :],
            need_weights=False,
            is_causal=True,
        )[0]


class SeqFirstSelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[S B H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=0.0,
            batch_first=False,
            bias=cfg_attention.bias_in_proj,
            add_bias_kv=cfg_attention.qkv_bias,
        )

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask[0, 0, :, :],
            need_weights=False,
            is_causal=True,
        )[0]


class SeqFirstSelfAttention(torch.nn.MultiheadAttention):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks.

    Inherits from MultiheadAttention to catch the same initialization
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[S B H]"

    def __init__(self, hidden_size: int, cfg_attention, norm_module=torch.nn.Identity):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        self.cfg_attention = cfg_attention
        self.use_fire = False

        self.norm = norm_module()

        # Strided linear layer.
        self.in_proj_weight = torch.nn.Parameter(torch.randn(3 * self.hidden_size, self.hidden_size))
        if cfg_attention.qkv_bias:
            self.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * self.hidden_size))
        else:
            self.in_proj_bias = None
        self.bias_k, self.bias_v = None, None  # for compat with MultiheadAttention

        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding == "sanity":
            self.rotary_emb = RotarySanityCheck(self.hidden_per_head, seq_dim=0)
        elif cfg_attention.rotary_embedding == "v2":
            self.rotary_emb = RotaryEleutherAI(self.hidden_per_head)
        elif cfg_attention.rotary_embedding == "llama":
            self.rotary_emb = RotaryLLAMA(self.hidden_per_head)
        elif cfg_attention.rotary_embedding == "fire":
            self.rotary_emb = FIRE(cfg_attention.num_attention_heads, max_length=cfg_attention.max_length)
            self.use_fire = True
        elif cfg_attention.rotary_embedding:
            self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            self.rotary_emb = None
            
        if cfg_attention.sequence_op == "torch-softmax":
            self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "shaped-attention":
            self.sequence_op = TorchShaped(cfg_attention.seq_op_in_fp32, hidden_size=self.hidden_size)
        elif cfg_attention.sequence_op == "swin-cosine":
            self.sequence_op = SwinCosine(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "torch-norm":
            self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "none":
            self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsum":
            self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsumexp":
            self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
        else:
            raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        if cfg_attention.skip_output_projection:
            self.out_proj = torch.nn.Identity()
        else:
            self.out_proj = NonDynamicallyQuantizableLinear(hidden_size, hidden_size, bias=cfg_attention.bias_in_proj)

        self.attention_func = self.attention

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False, fire: Optional[torch.Tensor] = None):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # this better be fused in a clever way:
        matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2)) * self.norm_factor

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        if fire is not None:
            attention_scores += fire

        # ===========================
        # Attention probs
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.sequence_op(attention_scores, attention_mask)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer
    
    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = torch.nn.functional.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        # new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_per_head)
        mixed_x_layer = mixed_x_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
        )
        # print("mixed shape ",mixed_x_layer.shape) (82, 24, 16, 192)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)
        
        fire = None
        if self.rotary_emb is not None:
            if self.use_fire:
                fire = self.rotary_emb(query_layer.size(0), query_layer.device)
            else:
                query_layer, key_layer = self.rotary_emb(query_layer, key_layer)
                # print(query_layer.shape)

        # ==================================
        # Attention computation
        # ==================================
        context_layer = self.attention_func(query_layer, key_layer, value_layer, attention_mask, self.training, fire)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        return self.out_proj(self.norm(context_layer))


class FourierMixing(torch.nn.Module):
    """Fourier mixing layer as described in the FNet paper.
    Layer takes input with size [Batch, Seq, Hidden] and returns output of the same size.
    This function can take an attention mask as input, but will ignore it.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.fft_op_in_fp32 = True  # Always necessary (atleast on pytorch 1.12)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(hidden_size, seq_dim=1))
            else:
                self.rotary_emb = Rotary(hidden_size, seq_dim=0)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Forward will take an attention mask but ignore it!"""

        if self.rotary_emb is not None:
            # full rotary (mostly on for compatibility, no guarantees on this being non-terrible)
            cos, sin = self.rotary_emb.get_cos_sin_cache(hidden_states)
            hidden_states = (hidden_states * cos[:, 0]) + (self.rotary_emb.rotate_half(hidden_states) * sin[:, 0])

        if self.fft_op_in_fp32:
            hidden_state_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
        else:
            hidden_state_dtype = None

        # Implementation 1:
        # hidden_states = torch.fft.fft(torch.fft.fft(hidden_states, dim=0, , norm="ortho"), dim=2, , norm="ortho").real
        # Implementation 2:
        hidden_states = torch.fft.fftn(hidden_states, dim=(1, 2), norm="ortho").real  # could also cast into angle?

        if self.fft_op_in_fp32:
            hidden_states = hidden_states.to(hidden_state_dtype)

        return hidden_states


class TorchSoftmax(torch.nn.Module):
    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs.masked_fill_(attention_mask, -10000.0)
        probs = torch.softmax(inputs, dim=-1).to(dtype=input_dtype)
        return probs


class TorchShaped(torch.nn.Module):
    """Noci et al."""

    def __init__(self, seq_op_in_fp32=False, hidden_size=768):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.register_buffer("nfactor", torch.tensor(hidden_size).rsqrt())

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        breakpoint()
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs.masked_fill_(attention_mask, -10000.0)
        probs = torch.softmax(inputs * self.nfactor, dim=-1).to(dtype=input_dtype)
        I = torch.eye(probs.shape[-1], dtype=probs.dtype, device=probs.device)[None, None, :, :]
        shaped_outputs = probs + I - 1 / probs.shape[-1]
        return shaped_outputs


class SwinCosine(torch.nn.Module):
    """kind of SwinCosine, but not quite (normalizations scaled by mean(q) and mean(k))"""

    def __init__(self, seq_op_in_fp32=False, tau=0.1, eps=1e-8):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.tau = 0.1
        self.eps = eps

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """inputs are q_i, k_j -> o_ij. Normalize"""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        row_norm = inputs.mean(dim=-1, keepdim=True).norm(dim=-2, keepdim=True)
        col_norm = inputs.mean(dim=-2, keepdim=True).norm(dim=-1, keepdim=True)
        outputs = inputs / torch.clamp(row_norm * col_norm * self.tau, min=self.eps)

        if attention_mask is not None:
            outputs[:, :, attention_mask[0, 0]] = 0

        return outputs.to(dtype=input_dtype)


class TorchNormalize(torch.nn.Module):
    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)

        if attention_mask is not None:
            inputs.masked_fill_(attention_mask, 0.0)

        norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms


class ScaledIdentity(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs * torch.as_tensor(inputs.shape[2]).rsqrt()).to(dtype=input_dtype)


class Cumsum(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.cumsum(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)


class CumsumExp(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = True  # Required as of pytorch 1.13

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.logcumsumexp(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)

"""Non-standard embedding implementations."""

import torch
import math

from typing import Tuple
from einops import repeat
import random


class PositionalEmbedding(torch.nn.Module):
    # https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15C1-L31C37
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = (1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))).float()
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        # sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        tensor_24_17_1 = pos_seq.float().unsqueeze(2)

        vector_512_expanded = self.inv_freq.unsqueeze(0).unsqueeze(1)

        result = torch.matmul(tensor_24_17_1, vector_512_expanded)

        sinusoid_inp = result.squeeze(2)

        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class RandomNoise(torch.nn.Module):

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        return torch.normal(0, 0.1, size=(input_ids.size(0), input_ids.size(1), self.embedding_dim)).to(input_ids.device)


class RPE(torch.nn.Module):
    # https://jaketae.github.io/study/relative-positional-encoding/
    # def __init__(self, embedding_dim, max_seq_length=5000):
    #     super().__init__()

    # def forward(self, input_ids):
    #     return torch.normal(0, 0.1, size=input_ids.shape)
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.query = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.Er = torch.nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError("sequence length exceeds model capacity")

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = torch.nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


# module partially stolen from pytorch examples:
class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.pe[:, : input_ids.shape[1], :]


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(torch.tensor([1.0 / embedding_dim**0.5]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.scale_factor * self.pe[:, : input_ids.shape[1], :]


class LearnablePositional(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embedding_dim, max_seq_length=1024):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        position_ids = self.position_ids[:, : input_ids.shape[1]]
        return self.embedding(position_ids)


class LearnablePositionalRand(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embedding_dim, max_seq_length=1024):
        super().__init__()
        self.max_length = max_seq_length
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        seq_length = input_ids.shape[1]
        device = input_ids.device
        if seq_length > self.max_length:  # max length will be increased to max sequnece length if max length is short
            max_length = seq_length
        else:
            max_length = self.max_length
        position_ids = self.position_ids[:, : input_ids.shape[1]]
        position_ids = torch.sort(torch.randperm(max_length, dtype=torch.long, device=device)[:seq_length]).values
        return self.embedding(position_ids)

# Code stolen from GPT-X:
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

        # Force fusions on batched version
        def rotate_half(x: torch.Tensor):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster
            return torch.cat((-x2, x1), dim=-1)

        def rope_fn(cos: torch.Tensor, sin: torch.Tensor, query_layer: torch.Tensor, key_layer: torch.Tensor):
            QK = torch.cat([query_layer, key_layer], dim=1)
            rotated = QK * cos[: QK.shape[0]] + rotate_half(QK) * sin[: QK.shape[0]]
            return torch.split(rotated, query_layer.shape[1], dim=1)

        self.rope_fn = rope_fn  # handle fusion on module level

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        cos_cached, sin_cached = self.get_cos_sin_cache(query_layer)
        return self.rope_fn(cos_cached, sin_cached, query_layer, key_layer)

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster

class RotarySanityCheck(torch.nn.Module):
    """not again..."""

    def __init__(self, dim, base=10000, def_seq_length=128, seq_dim: int = 0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.seq_len_cached = def_seq_length
        self.seq_dim = seq_dim
        cos_cache, sin_cache = self._get_cos_sin()
        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

    @torch.no_grad()
    def get_cos_sin_cache(self, x: torch.Tensor):
        seq_len = x.shape[self.seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = x.shape[self.seq_dim]
            cos_cache, sin_cache = self._get_cos_sin()
            self.cos_cached = cos_cache.to(x.device)
            self.sin_cached = sin_cache.to(x.device)
        return self.cos_cached, self.sin_cached

    def _get_cos_sin(self):
        t = torch.arange(self.seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.seq_dim == 0:
            return emb.cos()[:, None, None, :].detach(), emb.sin()[:, None, None, :].detach()
        else:
            return emb.cos()[None, :, None, :].detach(), emb.sin()[None, :, None, :].detach()

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        # cos, sin = self.get_cos_sin_cache(key_layer)
        # cos, sin = (cos[offset : query_layer.shape[0] + offset, ...], sin[offset : query_layer.shape[0] + offset, ...])
        cos, sin = self.cos_cached, self.sin_cached
        return (query_layer * cos) + (self.rotate_half(query_layer) * sin), (key_layer * cos) + (self.rotate_half(key_layer) * sin)

    def rotate_half(self, x: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # torch.split(x, x.shape[-1] // 2, dim=-1)  # not faster

    @torch.jit.export
    def single_forward(self, inputs: torch.Tensor):
        """For cases where shapes of Q and K do not match."""
        cos, sin = self.cos_cached[: inputs.shape[0]], self.sin_cached[: inputs.shape[0]]
        return inputs * cos + self.rotate_half(inputs) * sin


# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/rotary.py who adapted from
# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
class RotaryEleutherAI(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    """

    _seq_len_cached: int
    # _cos_cached: Optional[torch.Tensor]
    # _sin_cached: Optional[torch.Tensor]

    def __init__(self, dim_model: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)

        _cos_cached, _sin_cached = self._update_cos_sin_tables(torch.randn(1, 128, 1), seq_dimension=-2)
        self.register_buffer("_cos_cached", _cos_cached, persistent=False)
        self.register_buffer("_sin_cached", _sin_cached, persistent=False)

    @torch.jit.ignore
    def _update_cos_sin_tables(self, x: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        # if seq_len != self._seq_len_cached:  # or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
        self._seq_len_cached = seq_len
        t = torch.arange(x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype)
        # Don't do einsum, it converts fp32 to fp16
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos_cached = repeat(torch.cos(freqs).to(x.dtype), "... d -> ... (d 2)")
        sin_cached = repeat(torch.sin(freqs).to(x.dtype), "... d -> ... (d 2)")

        return cos_cached, sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        # self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=seq_dimension)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dimension),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dimension),
        )

    def rotate_half(self, x: torch.Tensor):
        x = x.unflatten(dim=-1, sizes=(-1, 2))
        x1, x2 = x.unbind(dim=-1)
        rotated_x = torch.stack((-x2, x1), dim=-1)
        return rotated_x.flatten(start_dim=-2)

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dimension: int = -2):
        # NOTE: This could probably be moved to Triton

        # Handle a possible sequence length mismatch in between q and k
        cos = cos[: x.shape[seq_dimension], :]
        sin = sin[: x.shape[seq_dimension], :]
        if seq_dimension == -3:
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        return (x * cos) + (self.rotate_half(x) * sin)


class RotaryLLAMA(torch.nn.Module):
    """Facebook implementation of rotary embeddings."""

    def __init__(self, hidden_per_head, base=10000, max_seq_length=512, seq_dim: int = 0):
        super().__init__()
        self.seq_dim: int = seq_dim
        freqs_cis = self.precompute_freqs_cis(dim=hidden_per_head, end=max_seq_length * 2, theta=base)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        return self.apply_rotary_emb(query_layer, key_layer, freqs_cis=self.freqs_cis)

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        freqs_cis = freqs_cis[: x.shape[self.seq_dim]]
        # shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        # shape = [1, seq_length, 1, hidden_per_head]
        shape = [s if i == self.seq_dim or i == x.ndim - 1 else 1 for i, s in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

class FIRE(torch.nn.Module):
    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-6, max_length=0):
        """
        FIRE attention bias module (https://arxiv.org/abs/2310.04418).

        Args:
            num_heads: number of attention heads.
            mlp_width: Width of MLP.
            init_c: initial value of log transformation parameter
            init_L: initial value of thresholding parameter
            eps: small constant for numerical stability
        """
        super(FIRE, self).__init__()
        self.max_length = max_length  # using random PE

        # Define the MLP layers
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, mlp_width), torch.nn.ReLU(), torch.nn.Linear(mlp_width, num_heads))

        # Initialize c (log transformation parameter)
        self.c = torch.nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = torch.nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = torch.nn.Parameter(torch.tensor(1.0))  # learn a multiplier to L

        self.eps = eps

    def forward(self, seq_length, device):
        """
        Compute FIRE attention bias (https://arxiv.org/abs/2310.04418).

        Args:
            x: input sequence, shape [bsz, num_heads, seq_len, hidden_dim]

        Returns:
            attention bias of shape [1, num_heads, seq_len, seq_len]
        """
        if (seq_length > self.max_length) or (
            not self.training
        ):  # max length will be increased to max sequnece length if max length is short
            max_length = seq_length
        else:
            max_length = self.max_length

        # take a subset (of length seq_length) of a random permutation of length max_length, then sort it to
        positions = torch.sort(torch.randperm(max_length, dtype=torch.float, device=device)[:seq_length]).values
        relative_distances = positions[:, None] - positions[None, :]
        
        # Thresholding the normalizer for short sequence modeling
        threshold = torch.abs(self.L_multiplier * self.init_L)
        position_normalizer = torch.max(positions, threshold)[:, None]

        # Amplifying differences among local positions with log transform
        relative_distances = torch.log(torch.abs(self.c * relative_distances) + 1)
        position_normalizer = torch.log(torch.abs(self.c * position_normalizer) + 1)

        # Progressive interpolation
        normalized_distances = relative_distances / (position_normalizer + self.eps)
        fire_bias = self.mlp(normalized_distances.unsqueeze(-1)).unsqueeze(0)
        fire_bias = fire_bias.permute(0, 3, 1, 2)
        
        return fire_bias

class Abacus(torch.nn.Module):
    """Abacus Embeddings, learned emebddings resued for each digit"""

    def __init__(self, embedding_dim, max_seq_length=1024, max_k=99):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))
        self.max_k = max_k # the max_k here by default is 99 as we add it on after istead of generate with it

    def helper(self, mask, device):
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids):
        """This is a batch-first implementation"""
        """
        This is a batch-first implementation
        designed to work with our tokenizers, for a more versatile implementation, look at the abacus.py file
        sort tokenizer: '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13

        {'0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13, 'D': 14, ',': 15, ':': 16, '=': 17, ' ': 18, 'A': 19, 'B': 20, 'C': 21, 'E': 22, 'F': 23, 'G': 24, 'H': 25, 'I': 26, 'J': 27, 'K': 28, 'L': 29, 'M': 30, 'N': 31, 'O': 32, 'P': 33, 'Q': 34, 'R': 35, 'S': 36, 'T': 37, 'U': 38, 'V': 39, 'W': 40, 'X': 41, 'Y': 42, 'Z': 43, 'a': 44, 'b': 45, 'c': 46, 'd': 47, 'e': 48, 'f': 49, 'g': 50, 'h': 51, 'i': 52, 'j': 53, 'k': 54, 'l': 55, 'm': 56, 'n': 57, 'o': 58, 'p': 59, 'q': 60, 'r': 61, 's': 62, 't': 63, 'u': 64, 'v': 65, 'w': 66, 'y': 67, 'z': 68, '!': 69, '@': 70, '£': 71, '#': 72, '$': 73, '%': 74, '^': 75, '&': 76, '*': 77, '(': 78, ')': 79, '~': 80, '?': 81, '.': 82, '<': 83, '>': 84, '{': 85, '}': 86, '[': 87, ']': 88, ';': 89, '/': 90, '|': 91, 'β': 92, 'Γ': 93, 'Δ': 94, 'δ': 95, 'ε': 96, 'ζ': 97, 'η': 98, 'θ': 99, 'κ': 100, 'Λ': 101, 'λ': 102, 'μ': 103, 'Ξ': 104, 'ξ': 105, 'Π': 106, 'π': 107, 'Σ': 108, 'ς': 109, 'τ': 110, 'Φ': 111, 'φ': 112, 'χ': 113, 'Ψ': 114, 'ψ': 115, 'Ω': 116, 'ω': 117, '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3}
        """
        mask = (input_ids >= 4) & (input_ids <= 13)
        output = self.helper(mask, input_ids.device)
        
        k=0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

        return self.embedding(output)
"""Basic transformer components."""

import torch

from typing import Tuple
from functools import partial

from .embeddings import SinusoidalPositional, LearnablePositional, ScaledSinosoidal, Abacus
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear  # use to mark output projections of attn while it exists

INPLACE = False


class EmbeddingComponent(torch.nn.Module):
    """Absolute Embeddings and sine embeddings"""
    def __init__(self, cfg_embedding, norm, norm_eps):
        super().__init__()

        self.word_embedding = torch.nn.Embedding(cfg_embedding.vocab_size, cfg_embedding.embedding_dim)
        if cfg_embedding.pos_embedding == "learned":
            self.pos_embedding = LearnablePositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "learned_rand":
            self.pos_embedding = LearnablePositionalRand(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "sinusoidal":
            self.pos_embedding = SinusoidalPositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "scaled-sinusoidal":
            self.pos_embedding = ScaledSinosoidal(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "abacus":
            self.pos_embedding = Abacus(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length, max_k=cfg_embedding.max_abacus_len)
        else:
            self.pos_embedding = None

        if cfg_embedding.normalization:
            self.stabilize_low_precision = cfg_embedding.get("stable_low_precision", False)
            self.norm = _get_norm_fn(norm)(cfg_embedding.embedding_dim, eps=norm_eps)
        else:
            self.stabilize_low_precision = False
            self.norm = torch.nn.Identity()

    def forward(self, input_ids):
        embeds = self.word_embedding(input_ids)

        if self.pos_embedding is not None:
            embeds += self.pos_embedding(input_ids)
        

        if self.stabilize_low_precision:
            # Stabilize as in bnb StableEmbedding
            return self.norm(embeds.to(torch.get_default_dtype())).to(embeds.dtype)
        else:
            return self.norm(embeds)


class PredictionHeadComponent(torch.nn.Module):
    def __init__(self, cfg_arch):
        super().__init__()

        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            output_size = cfg_arch.hidden_size
        else:
            output_size = cfg_arch.embedding.embedding_dim

        self.dense = torch.nn.Linear(cfg_arch.hidden_size, output_size, bias=cfg_arch.use_bias)
        self.nonlin = _get_nonlin_fn(cfg_arch.nonlin, use_gating=False)()
        self.norm = _get_norm_fn(cfg_arch.norm)(output_size, eps=cfg_arch.norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.norm(self.nonlin(self.dense(hidden_states)))
        return hidden_states


class NormalizedResidualConnection(torch.nn.Module):
    """Implement variations on residual connection types, especially stabilized versions and deep/shaped propagation."""

    def __init__(self, input_dim, cfg_arch, output_dim=None, dropout=0.0):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        if cfg_arch.norm_scheme == "pre":
            self.norm = _get_norm_fn(cfg_arch.norm)(input_dim, eps=cfg_arch.norm_eps)
            self._chosen_forward_impl = self._prenormalization_residual
        elif cfg_arch.norm_scheme == "post":
            self.norm = _get_norm_fn(cfg_arch.norm)(output_dim, eps=cfg_arch.norm_eps)
            self._chosen_forward_impl = self._postnormalization_residual
        elif cfg_arch.norm_scheme == "simple":
            self._chosen_forward_impl = self._simple_residual
        elif cfg_arch.norm_scheme == "deepnorm":
            self.norm = _get_norm_fn(cfg_arch.norm)(output_dim, eps=cfg_arch.norm_eps)
            if "num_transformer_layers" in cfg_arch:
                self.alpha = (2.0 * cfg_arch.num_transformer_layers) ** 0.25
            elif "layers_in_recurrent_block" in cfg_arch:
                self.alpha = (2.0 * cfg_arch.layers_in_recurrent_block * cfg_arch.maximal_recurrence) ** 0.25
            else:
                raise ValueError("Need to define `num_transformer_layers` in config for deepnorm.")
            self._chosen_forward_impl = self._deepnorm_residual
        elif cfg_arch.norm_scheme == "shaped":
            self.norm = _get_norm_fn(cfg_arch.norm)(input_dim, eps=cfg_arch.norm_eps)
            self.gamma = 0.214  # Noci et al., could make this into a parameter
            self.alpha = torch.as_tensor(1 - self.gamma**2).sqrt().item()
            self._chosen_forward_impl = self._prenorm_equalized_residual
        elif cfg_arch.norm_scheme == "sandwich":
            self.norm = _get_norm_fn(cfg_arch.norm)(input_dim, eps=cfg_arch.norm_eps)
            self.norm2 = _get_norm_fn(cfg_arch.norm)(output_dim, eps=cfg_arch.norm_eps)
            self._chosen_forward_impl = self._sandwich_residual
        else:
            raise ValueError(f"Invalid type of residual connection {cfg_arch.norm_scheme} given.")

    def _simple_residual(self, residual, layer, states, *args, **kwargs):
        return residual + self.dropout(layer(states, *args, **kwargs))

    def _prenormalization_residual(self, residual, layer, states, *args, **kwargs):
        return residual + self.dropout(layer(self.norm(states), *args, **kwargs))

    def _postnormalization_residual(self, residual, layer, states, *args, **kwargs):
        return self.norm(residual + layer(states, *args, **kwargs))

    def _deepnorm_residual(self, residual, layer, states, *args, **kwargs):
        return self.norm(residual * self.alpha + self.dropout(layer(states, *args, **kwargs)))

    def _prenorm_equalized_residual(self, residual, layer, states, *args, **kwargs):
        return residual * self.alpha + self.dropout(layer(self.norm(states), *args, **kwargs)) * self.gamma

    def _sandwich_residual(self, residual, layer, states, *args, **kwargs):
        return self.norm2(residual + self.dropout(layer(self.norm(states), *args, **kwargs)))

    def forward(self, residual: torch.Tensor, layer_callable: torch.nn.Module, states: torch.Tensor, *args, **kwargs):
        """Argument might look weird here, but I find it nicer because it reads like the pre/post schemes from left to right,
        as
        residual + layer ( state )

        Additional args are passed directly into the layer callable
        """
        return self._chosen_forward_impl(residual, layer_callable, states, *args, **kwargs)


def _get_norm_fn(norm_name):
    if norm_name == "ScaleNorm":
        norm_fn = ScaleNorm
    elif norm_name == "RMSNorm":
        norm_fn = RMSNorm
    elif norm_name == "ApexLayerNorm":
        from apex.normalization import FusedLayerNorm

        norm_fn = FusedLayerNorm
    else:
        norm_fn = getattr(torch.nn, norm_name)
    return norm_fn


def _get_nonlin_fn(nonlin_name, use_gating=True):
    if "glu" in nonlin_name.lower():
        nonlin_name = nonlin_name.split("glu")[0]
        wrap_in_glu = use_gating
    else:
        wrap_in_glu = False
    nonlin_fn = getattr(torch.nn, nonlin_name)  # dont mess this up :<
    try:
        nonlin_fn = partial(nonlin_fn, inplace=INPLACE)
        nonlin_fn()
    except TypeError:
        nonlin_fn = getattr(torch.nn, nonlin_name)

    if wrap_in_glu:
        return partial(GLU, nonlin_fn)
    else:
        return nonlin_fn


class GLU(torch.nn.Module):
    """*-GLU activation functions.

    Implementation mostly following megatron
    """

    def __init__(self, sub_activation):
        super().__init__()
        self.sub_activation = sub_activation()

    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        return self.sub_activation(gate) * x


class ScaleNorm(torch.nn.Module):
    """Quick and simple scale norm implementation. "elementwise_affine" is not the ideal name but for compat with LayerNorm

    Do we also need FixNorm (cosine in the last layer)? It's a maybe here:
    https://github.com/lucidrains/performer-pytorch/issues/55#issuecomment-762544686
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.learnable_scale = torch.nn.Parameter(torch.tensor(float(hidden_size) ** -0.5))
        else:
            self.register_buffer("learnable_scale", torch.tensor(float(hidden_size) ** -0.5))

    def forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=self.eps)


class RMSNorm(torch.nn.Module):
    """The RMS variant of scaling norms.  "elementwise_affine" is not the ideal name but for compat with LayerNorm"""

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.learnable_scale = torch.nn.Parameter(torch.ones(hidden_size) ** -0.5)
        else:
            self.register_buffer("learnable_scale", torch.ones(hidden_size) ** -0.5)

    def _legacy_forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=1e-8)

    def _norm(self, x):
        """LLama implementation"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.learnable_scale


def get_causal_attention_mask(input_ids) -> torch.Tensor:
    """Simplified triangular causal mask. Adapted for multiple heads."""
    seq_length = input_ids.shape[1]  # not transposed yet
    device = input_ids.device
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, 1, seq_length, seq_length), device=device)).view(1, 1, seq_length, seq_length)

    # convert to binary
    return mask < 0.5


def get_extended_attention_mask(attention_mask: torch.Tensor, input_shape: Tuple[int], causal_attention: bool = False) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.

    Method stolen from huggingface :)
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if causal_attention:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=attention_mask.device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones((batch_size, seq_length, prefix_seq_len), device=attention_mask.device, dtype=causal_mask.dtype),
                        causal_mask,
                    ],
                    axis=-1,
                )
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    # extended_attention_mask = extended_attention_mask.to(dtype=self.setup["dtype"])  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


"""Collect inits."""


@torch.no_grad()
def _init_module(module, init_method="normal", init_std=0.02, hidden_size=768, num_layers=12):
    """Todo: refactor this insanity"""
    if "deepnorm" in init_method:  # This is a xavier init with changes in the MHA inits
        if "normal" in init_method:
            gain = init_std
        elif "subln" in init_method:
            gain = torch.as_tensor(2 * num_layers).log().sqrt()  # foundation transformer paper, use with subln
        elif "straight" in init_method:
            gain = torch.as_tensor(8 * num_layers).pow(-0.25)  # deepnorm paper, use with deepnorm
        elif "as-is" in init_method:  # use locally defined inits for each module
            gain = 1.0
        else:
            raise ValueError(f"Invalid init method {init_method} given.")

        if isinstance(module, torch.nn.Linear):
            if isinstance(module, NonDynamicallyQuantizableLinear):
                # This is handled below in the MultiheadAttention section
                pass
            else:
                if module.weight is not None:
                    torch.nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=module.weight.shape[1] ** -0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, torch.nn.MultiheadAttention):  # be careful with other transformer definitions!
            if "mimetic" in init_method:
                if module.in_proj_weight is not None:
                    h = module.in_proj_weight.shape[1]
                    Z1 = module.in_proj_weight.new_empty([h, h])
                    torch.nn.init.xavier_normal_(Z1, gain=gain)  # as per deepnorm prescription
                    I = torch.eye(h, device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype)
                    U1, S1, V1 = torch.linalg.svd(Z1 + I, full_matrices=False)
                    V = U1 @ torch.diag_embed(S1.sqrt())
                    O = V1 @ torch.diag_embed(S1.sqrt())

                    k = module.head_dim
                    I = torch.eye(h, device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype)
                    Qlist, Klist = [], []
                    for head in range(module.num_heads):
                        Z2 = module.in_proj_weight.new_empty([h, h])
                        torch.nn.init.xavier_normal_(Z2, gain=1.0)  # as per deepnorm prescription
                        U2, S2, V2 = torch.linalg.svd(Z2 + I, full_matrices=False)
                        Qlist.append(U2[:, :k] @ torch.diag_embed(S2[:k].sqrt()))
                        Klist.append(V2[:, :k] @ torch.diag_embed(S2[:k].sqrt()))
                    Q, K = torch.cat(Qlist, dim=-1), torch.cat(Klist, dim=-1)
                    module.in_proj_weight.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())
                    if module.out_proj is not None:
                        module.out_proj.weight.data.copy_(O)
            else:
                if module.in_proj_weight is not None:
                    h = module.in_proj_weight.shape[1]
                    Q, K, V = (
                        module.in_proj_weight.new_empty([h, h]),
                        module.in_proj_weight.new_empty([h, h]),
                        module.in_proj_weight.new_empty([h, h]),
                    )
                    torch.nn.init.xavier_normal_(Q, gain=1.0)  # as per deepnorm prescription
                    torch.nn.init.xavier_normal_(K, gain=1.0)
                    torch.nn.init.xavier_normal_(V, gain=gain)
                    module.in_proj_weight.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())
                # init outproj:
                if module.out_proj is not None:
                    torch.nn.init.xavier_normal_(module.out_proj.weight, gain=gain)
                    if module.out_proj.bias is not None:
                        module.out_proj.bias.data.zero_()
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()
            if module.out_proj is not None and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
    else:
        if "normal" in init_method:
            std = init_std
        elif init_method == "small" in init_method:
            # Transformers without Tears: Improving
            # the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010)
            std = torch.as_tensor(2 / (5 * hidden_size)).sqrt()
        elif "megatron" in init_method:
            std = torch.as_tensor(1 / (3 * hidden_size)).sqrt()
            # Megatron init is near-equal to normal if hidden=768, but otherwise smaller
        elif "wang" in init_method:
            std = 2 / num_layers / torch.as_tensor(hidden_size).sqrt()
        elif "as-is" in init_method:  # use locally defined inits for each module
            return
        else:
            raise ValueError(f"Invalid init method {init_method} given.")
        if isinstance(module, torch.nn.Linear):
            if isinstance(module, NonDynamicallyQuantizableLinear):
                # This is handled below in the MultiheadAttention section
                pass
            else:
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                if module.weight is not None:
                    module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        elif isinstance(module, torch.nn.MultiheadAttention):  # be careful with other transformer definitions!
            if "mimetic" in init_method:
                if module.in_proj_weight is not None:
                    h = module.in_proj_weight.shape[1]
                    Z1 = module.in_proj_weight.new_empty([h, h]).normal_() / h
                    I = torch.eye(h, device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype)
                    U1, S1, V1 = torch.linalg.svd(0.2 * Z1 + 0.2 * I, full_matrices=False)
                    V = U1 @ torch.diag_embed(S1.sqrt())
                    O = V1 @ torch.diag_embed(S1.sqrt())

                    k = module.head_dim
                    I = torch.eye(h, device=module.in_proj_weight.device, dtype=module.in_proj_weight.dtype)
                    Qlist, Klist = [], []
                    for head in range(module.num_heads):
                        # Z2 = module.in_proj_weight.new_empty([h, h]).normal_() / h
                        U2, S2, V2 = torch.linalg.svd(0 + 0.5 * I, full_matrices=False)  # alpha1 =0 from Trockman
                        Qlist.append(U2[:, :k] @ torch.diag_embed(S2[:k].sqrt()))  # this is a bit pointless, ...
                        Klist.append(V2[:, :k] @ torch.diag_embed(S2[:k].sqrt()))  # ... I've left it here for alpha1 not zero
                    Q, K = torch.cat(Qlist, dim=-1), torch.cat(Klist, dim=-1)
                    module.in_proj_weight.data.copy_(torch.cat([Q, K, V], dim=0).contiguous())
                    if module.out_proj is not None:
                        module.out_proj.weight.data.copy_(O)
            else:
                if module.in_proj_weight is not None:
                    module.in_proj_weight.data.normal_(mean=0.0, std=std)
                if module.out_proj is not None:
                    module.out_proj.weight.data.normal_(mean=0.0, std=std)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.bias_k is not None:
                module.bias_k.data.zero_()
            if module.bias_v is not None:
                module.bias_v.data.zero_()
            # init outproj:
            if module.out_proj is not None and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()

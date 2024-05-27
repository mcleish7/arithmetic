"""Base file for modifications of the transformer architecture"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from typing import Optional
from omegaconf import OmegaConf

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    NormalizedResidualConnection,
    EmbeddingComponent,
    GLU,
    get_causal_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism


class crammedTransformerConfig(PretrainedConfig):
    model_type = "crammedTransformer"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_transformer(cfg_arch, vocab_size):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size

    config = crammedTransformerConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    model = ScriptableLMForPreTraining(config)

    return model


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    Better do this manually and choose a sensible intermed_size that is nicely divisible.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, cfg_arch, output_size=None):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=cfg_arch.use_bias)
        self.nonlin = _get_nonlin_fn(cfg_arch.nonlin)()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        if cfg_arch.sub_normalization:
            self.norm = _get_norm_fn(cfg_arch.norm)(intermed_output_size, eps=cfg_arch.norm_eps)
        else:
            self.norm = torch.nn.Identity()
        output_size = hidden_size if output_size is None else output_size
        self.dense_out = torch.nn.Linear(intermed_output_size, output_size, bias=cfg_arch.use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.norm(self.nonlin(self.dense_in(hidden_states))))


class TransformerLayer(torch.nn.Module):
    """A transformer structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.residual1 = NormalizedResidualConnection(cfg_arch.hidden_size, cfg_arch)
        self.residual2 = NormalizedResidualConnection(cfg_arch.hidden_size, cfg_arch)
        if cfg_arch.attention.sub_normalization:
            sub_norm_fn = lambda: get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)  # noqa
        else:
            sub_norm_fn = torch.nn.Identity
        self.attn = get_attention_mechanism(idx, cfg_arch.hidden_size, cfg_arch.attention, sub_norm_fn)
        self.ffn = FFNComponent(cfg_arch.hidden_size, cfg_arch.intermed_size, cfg_arch)
        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = self.residual1(states, self.attn, states, attention_mask)
        states = self.residual2(states, self.ffn, states)
        return states


class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        self.register_buffer("attention_mask", torch.ones([0, 0, 0, 0], dtype=torch.bool), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        if input_ids.shape[1] != self.attention_mask.shape[1]:
            self.attention_mask = get_causal_attention_mask(input_ids)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, self.attention_mask)

        # if self.seq_first:
        #     hidden_states = hidden_states.transpose(0, 1).contiguous()
        # this happens only in the output if necessary

        return self.final_norm(hidden_states)


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        outputs = self.decoder(self.encoder(input_ids))

        if self.encoder.seq_first:
            shifted_outputs = outputs[:-1]
            shifted_labels = input_ids.transpose(0, 1)[1:].contiguous()
            outputs = outputs.detach().transpose(0, 1)
        else:
            shifted_outputs = outputs[..., :-1, :].contiguous()
            shifted_labels = input_ids[..., 1:].contiguous()
            outputs = outputs.detach()
        # Flatten the tokens and compute loss
        loss = self.loss_fn(shifted_outputs.view(-1, shifted_outputs.shape[-1]), shifted_labels.view(-1))

        return {"loss": loss, "logits": outputs[:, -1, :], "log_perplexity": loss.clone().detach()}


# ###### HF registry here? ############### #

AutoConfig.register("crammedTransformer", crammedTransformerConfig)
AutoModel.register(crammedTransformerConfig, ScriptableLM)
AutoModelForCausalLM.register(crammedTransformerConfig, ScriptableLMForPreTraining)

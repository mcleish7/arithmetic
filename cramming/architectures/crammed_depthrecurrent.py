"""Variant for modifications of the transformer architecture that are depth-recurrent"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from typing import Optional
from omegaconf import OmegaConf

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    GLU,
    get_causal_attention_mask,
    _init_module,
    NormalizedResidualConnection,
)
from .attention import get_attention_mechanism


class crammedDepthRecurrentConfig(PretrainedConfig):
    model_type = "crammedDepthRecurrent"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_recurrent(cfg_arch, vocab_size, equals_token):
    """See the config file for details on what is possible."""
    cfg_arch.embedding.vocab_size = vocab_size

    config = crammedDepthRecurrentConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    if config.arch["objective_layout"] in ["fixed", "albert"]:
        model = ScriptableRecurrentLMForPreTraining(config)
    elif config.arch["objective_layout"] in ["TBPTT", "deepthinking"]:
        model = ScriptableRecurrentLMBPTT(config, equals_token)
    else:
        raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")

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
            sub_norm_fn = lambda: _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)  # noqa
        else:
            sub_norm_fn = torch.nn.Identity
        self.attn = get_attention_mechanism(idx, cfg_arch.hidden_size, cfg_arch.attention, sub_norm_fn)
        self.ffn = FFNComponent(cfg_arch.hidden_size, cfg_arch.intermed_size, cfg_arch)
        self.LAYOUT = self.attn.LAYOUT

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = self.residual1(states, self.attn, states, attention_mask)
        states = self.residual2(states, self.ffn, states)
        return states


class TransformerBlock(torch.nn.Module):
    """A transformer block of multiple layers (without weightsharing)."""

    def __init__(self, layers, cfg_arch):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.injection_type = cfg_arch.input_injection_type
        if self.injection_type == "linear":
            self.adapter = torch.nn.Linear(cfg_arch.hidden_size * 2, cfg_arch.hidden_size, bias=False)
        elif self.injection_type == "ffn":
            self.ffn = FFNComponent(cfg_arch.hidden_size * 2, cfg_arch.intermed_size, cfg_arch, cfg_arch.hidden_size)

    def forward(self, states, injected_state, attention_mask: Optional[torch.Tensor] = None):
        if self.injection_type == "none":
            states = states
        elif self.injection_type == "add": # this is the deafault in the config
            states = states + injected_state
        elif self.injection_type == "linear":
            combined_inputs = torch.cat([states, injected_state], dim=-1)
            states = self.adapter(combined_inputs)
        elif self.injection_type == "ffn":
            combined_inputs = torch.cat([states, injected_state], dim=-1)
            states = self.ffn(combined_inputs)
        for layer in self.layers:
            states = layer(states, attention_mask)
        return states


class TransposedAdapter(torch.nn.Linear):  # steal init
    def __init__(self, embedding_dim, hidden_size, original_adapter, tie_weights=True):
        torch.nn.Module.__init__(self)
        # self.adapter.weight = self.encoder.adapter.weight.T # this would be nice but cannot assign like this
        if tie_weights:
            self.weight = original_adapter.weight
        else:
            self.adapter_active = False
            self.weight = torch.nn.Parameter(torch.randn([hidden_size, embedding_dim]))  # transposed
        self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight.T)


class ScriptableRecurrentLM(PreTrainedModel):
    """Depth-recurrent model. Trying to include most reasonable variations of this concept"""

    config_class = crammedDepthRecurrentConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        if self.cfg.embedding.embedding_dim != self.cfg.hidden_size:
            self.adapter = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.hidden_size, bias=False)
        else:
            self.adapter = torch.nn.Identity()
        self.state_init = self.cfg.state_init
        self.recurrent_block = torch.compile(
            TransformerBlock([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.layers_in_recurrent_block)], self.cfg),
            mode="default",
            disable=not self.cfg.local_compilation,
        )
        self.seq_first = self.recurrent_block.seq_first
        if self.cfg.head == "identity":
            self.head = torch.nn.Identity()
        elif self.cfg.head == "ffn":
            self.head = FFNComponent(self.cfg.hidden_size, self.cfg.intermed_size, self.cfg)
        elif self.cfg.head == "linear":
            self.head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.use_bias)
        else:
            raise ValueError(f"Invalid head layout {self.cfg.head} given.")

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        self.register_buffer("attention_mask", torch.ones([0, 0, 0, 0], dtype=torch.bool), persistent=False)

    def forward(self, input_ids: torch.Tensor, num_steps_no_grad: int = None, num_steps_with_grad: int = None):
        if input_ids.shape[1] != self.attention_mask.shape[1]:
            self.attention_mask = get_causal_attention_mask(input_ids)
        hidden_states = self.adapter(self.embedding(input_ids))
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        injected_state = hidden_states.clone()

        num_steps_prefix = 0 if num_steps_no_grad is None else num_steps_no_grad
        hidden_states = self.initialize_state(hidden_states)

        # Recurr without gradients
        with torch.no_grad():
            for repeat in range(num_steps_prefix):
                hidden_states = self.recurrent_block(hidden_states, injected_state, self.attention_mask).clone()

        num_steps_active = self.cfg.maximal_recurrence if num_steps_with_grad is None else num_steps_with_grad
        # Recur with gradients
        for repeat in range(num_steps_active):
            hidden_states = self.recurrent_block(hidden_states, injected_state, self.attention_mask).clone()
        return self.final_norm(self.head(hidden_states))

    def initialize_state(self, hidden_states):
        if self.cfg.initial_hidden_randomized:
            batch_size = hidden_states.shape[0]
            if self.state_init == "normal":
                hidden_states = torch.randn_like(hidden_states)
            elif self.state_init == "embed":  # initialized like a BERT embedding
                hidden_states = torch.randn_like(hidden_states).mul(0.02)
            elif self.state_init == "zero":
                hidden_states = torch.zeros_like(hidden_states)
            elif self.state_init == "unit":
                hidden_states = torch.randn_like(hidden_states)
                std, mean = torch.std_mean(hidden_states, dim=-1, keepdim=True)
                hidden_states = (hidden_states - mean) / std
        return hidden_states


class ScriptableRecurrentLMReplicaConcat(PreTrainedModel):
    """Depth-recurrent model. with skips inside block 
    This is nearly the same as ScriptableRecurrentLM but has skips inside block too"""

    config_class = crammedDepthRecurrentConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        if self.cfg.embedding.embedding_dim != self.cfg.hidden_size:
            self.adapter = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.hidden_size, bias=False)
        else:
            self.adapter = torch.nn.Identity()
        self.state_init = self.cfg.state_init


        self.max_recurs = self.cfg.layers_in_recurrent_block
        self.recurrent_blocks = []
        print("Initializing feedforward blocks with recall connections")
        for _ in range(self.max_recurs):
            self.recurrent_blocks.append(
                torch.compile(TransformerBlock([TransformerLayer(1, self.cfg)], self.cfg),
                              mode="default",
                              disable=not self.cfg.local_compilation,)
            )
        self.recurrent_blocks = torch.nn.ModuleList(self.recurrent_blocks)
        print(f"Initialized feedforward blocks with recall connections. "
              f"It has the depth of {self.max_recurs}")

        self.seq_first = self.recurrent_blocks[0].seq_first
        if self.cfg.head == "identity":
            self.head = torch.nn.Identity()
        elif self.cfg.head == "ffn":
            self.head = FFNComponent(self.cfg.hidden_size, self.cfg.intermed_size, self.cfg)
        elif self.cfg.head == "linear":
            self.head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.use_bias)
        else:
            raise ValueError(f"Invalid head layout {self.cfg.head} given.")

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        self.register_buffer("attention_mask", torch.ones([0, 0, 0, 0], dtype=torch.bool), persistent=False)


    def apply_recurrent_block(self, hidden_states, injected_state, attention_mask):
        for block in self.recurrent_blocks:
            hidden_states = block(hidden_states, injected_state, attention_mask)
        return hidden_states


    def forward(self, input_ids: torch.Tensor, num_steps_no_grad: int = None, num_steps_with_grad: int = None):
        if input_ids.shape[1] != self.attention_mask.shape[1]:
            self.attention_mask = get_causal_attention_mask(input_ids)
        hidden_states = self.adapter(self.embedding(input_ids))
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        injected_state = hidden_states.clone()

        num_steps_prefix = 0 if num_steps_no_grad is None else num_steps_no_grad
        hidden_states = self.initialize_state(hidden_states)

        # Recurr without gradients
        with torch.no_grad():
            for repeat in range(num_steps_prefix):
                hidden_states = self.apply_recurrent_block(hidden_states, injected_state, self.attention_mask).clone()

        num_steps_active = self.cfg.maximal_recurrence if num_steps_with_grad is None else num_steps_with_grad
        # Recur with gradients
        for repeat in range(num_steps_active):
            hidden_states = self.apply_recurrent_block(hidden_states, injected_state, self.attention_mask).clone()
        return self.final_norm(self.head(hidden_states))

    def initialize_state(self, hidden_states):
        if self.cfg.initial_hidden_randomized:
            batch_size = hidden_states.shape[0]
            if self.state_init == "normal":
                hidden_states = torch.randn_like(hidden_states)
            elif self.state_init == "embed":  # initialized like a BERT embedding
                hidden_states = torch.randn_like(hidden_states).mul(0.02)
            elif self.state_init == "zero":
                hidden_states = torch.zeros_like(hidden_states)
            elif self.state_init == "unit":
                hidden_states = torch.randn_like(hidden_states)
                std, mean = torch.std_mean(hidden_states, dim=-1, keepdim=True)
                hidden_states = (hidden_states - mean) / std
        return hidden_states


"""Generator fn for these models."""
@torch.no_grad()
def _generate(self, input_ids, token_limit=100, temperature=1.0, steps_at_generation_time=None, track_steps=False, greedy=False, quick=False, **kwargs):
    """Generate token_limit many tokens from input_ids prompt. 
    track_steps = for making thinking plots
    """
    predicted_ids = []
    tracking = []
    num_steps = self.cfg.maximal_recurrence_in_eval if steps_at_generation_time is None else steps_at_generation_time
    logit_tensor = torch.zeros(token_limit, num_steps, self.cfg.embedding.vocab_size)
    for gen_idx in range(token_limit):
        if input_ids.shape[1] != self.encoder.attention_mask.shape[1]:
            self.encoder.attention_mask = get_causal_attention_mask(input_ids)
        hidden_states = self.encoder.adapter(self.encoder.embedding(input_ids))
        if self.encoder.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        injected_state = hidden_states
        hidden_states = self.encoder.initialize_state(hidden_states)
        # Recur without gradient
        step = []
        with torch.no_grad():
            for repeat in range(num_steps):
                if hasattr(self.encoder, 'recurrent_blocks'):
                    for block in self.encoder.recurrent_blocks:
                        hidden_states = block(hidden_states, injected_state, self.encoder.attention_mask)
                else:
                    hidden_states = self.encoder.recurrent_block._orig_mod(hidden_states, injected_state,
                                                                           self.encoder.attention_mask)
                if track_steps:
                    # keep track of the intermediate probs
                    output_states = self.encoder.final_norm(self.encoder.head(hidden_states.clone()))
                    logits = self.decoder(self.adapter(output_states))
                    logits = logits[-1, :, :] if self.encoder.seq_first else logits[:, -1, :]
                    if greedy:
                        probs = torch.softmax(logits, dim=-1)
                        predicted_token = torch.argmax(logits, dim=1).unsqueeze(dim=0)
                    else:
                        probs = torch.softmax(logits * temperature, dim=-1)
                        predicted_token = torch.multinomial(probs, 1)
                    logit_tensor[gen_idx, repeat, :] = probs
                    step.append(predicted_token)
        if track_steps:
            predicted_token = step[-1]
        else:
            # calcualte the probs if we haven't already
            output_states = self.encoder.final_norm(self.encoder.head(hidden_states.clone()))
            logits = self.decoder(self.adapter(output_states))
            logits = logits[-1, :, :] if self.encoder.seq_first else logits[:, -1, :]
            if greedy:
                predicted_token = torch.argmax(logits, dim=1).unsqueeze(dim=0)
            else:
                predicted_token = torch.multinomial(torch.softmax(logits * temperature, dim=-1), 1)

        if quick:
            input_ids = torch.cat((input_ids, torch.transpose(predicted_token, 0, 1)), dim=1)
        else:
            input_ids = torch.cat([input_ids, predicted_token], dim=-1)
        predicted_ids += [predicted_token]
        tracking.append(step)

    if quick:
        generated_ids = torch.stack(predicted_ids, dim=1).squeeze()
    else:
        generated_ids = torch.cat(predicted_ids, dim=-1)

    if track_steps:
        return generated_ids, tracking, logit_tensor # tracking is a [num generated tokens, num recurrences] list of lists of tensors of which each tensor is a token id
    return generated_ids


class ScriptableRecurrentLMForPreTraining(PreTrainedModel):
    """Pretraining version"""

    config_class = crammedDepthRecurrentConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableRecurrentLM(config)
        if self.cfg.embedding.embedding_dim != self.cfg.hidden_size:
            self.adapter = TransposedAdapter(
                self.cfg.embedding.embedding_dim, self.cfg.hidden_size, self.encoder.adapter, self.cfg.tie_weights
            )
        else:
            self.adapter = torch.nn.Identity()
        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        if self.cfg.tie_weights:
            self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # size_average defaults to True so when using masking loss is calculated correctly

        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.layers_in_recurrent_block * self.cfg.maximal_recurrence,
            )

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        outputs = self.decoder(self.adapter(self.encoder(input_ids, num_steps_no_grad=0, num_steps_with_grad=self.cfg.maximal_recurrence)))

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

    def _generate(self, input_ids, token_limit=100, temperature=0.7, steps_at_generation_time=None):
        return _generate(self, input_ids, token_limit, temperature, steps_at_generation_time)


class ScriptableRecurrentLMBPTT(PreTrainedModel):
    """Pretraining version with stochastic depth / trunc. BPTT"""

    config_class = crammedDepthRecurrentConfig

    def __init__(self, config, equals_token):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.equals_token = equals_token

        self.max_recurrences_for_training = self.cfg.maximal_recurrence
        self.max_backprop = max(self.cfg.maximal_recurrence // 2 if self.cfg.max_backprop is None else self.cfg.max_backprop, 1)
        try:
            self.forward_only_model_with_skip = self.cfg.forward_only_model_with_skip
            if self.cfg.forward_only_model_with_skip:
                print("Using forward only model with skip")
                self.encoder = ScriptableRecurrentLMReplicaConcat(config)
            else:
                self.encoder = ScriptableRecurrentLM(config)
        except:
            self.encoder = ScriptableRecurrentLM(config)

        self.adapter = TransposedAdapter(self.cfg.embedding.embedding_dim, self.cfg.hidden_size, self.encoder.adapter, self.cfg.tie_weights)
        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        if self.cfg.tie_weights:
            self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.throttle = self.cfg.throttle
        self.alpha = self.cfg.alpha
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction=self.cfg.loss_reduction) # size_average defaults to True so when using masking loss is calculated correctly
        self._init_weights()

        self.mask_before_equals = self.cfg.mask_before_equals
        self.model_call = self.prog_model_call_with_masking # moved the logic for masking before equals into this function

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.layers_in_recurrent_block * self.cfg.maximal_recurrence,
            )

    def set_max_recurrences_for_training(self, new_max):
        """Can play around with recurrences during training"""
        self.max_recurrences_for_training = new_max
        self.max_backprop = max(self.max_recurrences_for_training // 2 if self.cfg.max_backprop is None else self.cfg.max_backprop, 1)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        """
        WARNING: max iters outputs is used for logits and entropy calcs
        """
        if self.training:
            loss, outputs = self.forward_progressive(input_ids)
            if self.throttle:
                Ek = 1 + min(self.max_recurrences_for_training / 4, self.max_backprop / 2)
                loss = loss * (Ek / self.max_backprop)
        else:
            loss, outputs = self.model_call(input_ids, n=self.cfg.maximal_recurrence_in_eval, k=0)

        return {"loss": loss, "logits": outputs[:, -1, :], "log_perplexity": loss.clone().detach()}
    
    def forward_progressive(self, input_ids):
        """Implements progressive loss"""
        if self.alpha != 1:
            # max iters forward pass
            n = self.max_recurrences_for_training-self.max_backprop
            k = self.max_backprop # i.e. maxmimise the number of layers we back prop through
            loss_max_iters, outputs_max_iters = self.model_call(input_ids, n=n, k=k)
        else:
            loss_max_iters = torch.zeros(1, dtype=torch.float32).to(input_ids.get_device())

        if self.alpha != 0:
            # stochastic forward pass
            n = torch.randint(low=0, high=self.max_recurrences_for_training, size=(1,))
            k = torch.randint(low=1, high=1 + min(self.max_recurrences_for_training - n, self.max_backprop), size=(1,))
            loss_progressive, outputs_progressive = self.model_call(input_ids, n=n, k=k)
            if self.alpha == 1:
                outputs_max_iters = outputs_progressive
        else:
            loss_progressive = torch.zeros(1, dtype=torch.float32).to(input_ids.get_device())
        
        loss = (1 - self.alpha) * loss_max_iters + self.alpha * loss_progressive
        # Returning outputs max_iters to be used for logits, could try outputs_progressive
        return loss, outputs_max_iters

    def prog_model_call_with_masking(self, input_ids, n, k):
        if self.mask_before_equals: # mask before equals
            indices_of_equals = (input_ids == self.equals_token).nonzero()[:, 1] # gets the index of equals sign for each tensor in the batch
            max_indices = torch.arange(input_ids.size(1), device=input_ids.device) # tensor for mask
            masks = max_indices.unsqueeze(0) > indices_of_equals.unsqueeze(1) # fill tensor after including index of = sign for each row
        else: # mask only the random padding
            masks = input_ids != 0
        
        outputs = self.decoder(self.adapter(self.encoder(input_ids, num_steps_no_grad=n, num_steps_with_grad=k)))

        if self.encoder.seq_first:
            shifted_outputs = outputs[:-1]
            shifted_labels = input_ids.transpose(0, 1)[1:].contiguous()
            outputs = outputs.detach().transpose(0, 1)
            masked = torch.mul(shifted_labels, masks[..., 1:].transpose(0, 1))
        else:
            shifted_outputs = outputs[..., :-1, :].contiguous()
            shifted_labels = input_ids[..., 1:].contiguous()
            outputs = outputs.detach()
            masked = torch.mul(shifted_labels, masks[..., 1:])
        masked[masked == 0] = -100 # mask all 0's in loss

        shifted_outputs_shape = shifted_outputs.shape
        
        loss = self.loss_fn(shifted_outputs.view(-1, shifted_outputs.shape[-1]), masked.view(-1)) # CE_Loss(Input, Target)
        if self.cfg.loss_reduction=='none': # giving all output samples equal weighting
            loss = loss.view(shifted_outputs_shape[0],shifted_outputs_shape[1])
            loss = torch.mean(loss, dim=1)
            loss = torch.mean(loss)
        return loss, outputs

    def _generate(self, input_ids, token_limit=100, temperature=1.0, steps_at_generation_time=None, track_steps=False, greedy=False, quick=False):
        return _generate(self, input_ids, token_limit, temperature, steps_at_generation_time, track_steps, greedy=greedy, quick=quick)


# ###### HF registry here? ############### #

AutoConfig.register("crammedDepthRecurrent", crammedDepthRecurrentConfig)
AutoModel.register(crammedDepthRecurrentConfig, ScriptableRecurrentLM)
AutoModelForCausalLM.register(crammedDepthRecurrentConfig, ScriptableRecurrentLMForPreTraining)

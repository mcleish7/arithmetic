"""Interface to construct models."""

from .huggingface_interface import construct_huggingface_model
from .sanity_check import SanityCheckforPreTraining
from .crammed_transformer import construct_crammed_transformer
from .crammed_depthrecurrent import construct_crammed_recurrent

import logging
from ..utils import is_main_process

log = logging.getLogger(__name__)


def construct_model(cfg_arch, tokenizer):
    model = None
    eos_token_id = tokenizer.eos_token  # tokenizer.vocab["<eot>"]
    if "model_type" in cfg_arch:
        # attempt to solve locally
        if "SanityCheckLM" in cfg_arch.model_type:
            model = SanityCheckforPreTraining(cfg_arch.width, tokenizer.vocab_size)
        elif "ScriptableCrammedTransformer" in cfg_arch.model_type:
            model = construct_crammed_transformer(cfg_arch, tokenizer.vocab_size)
        elif "ScriptableCrammedDepthRecurrent" in cfg_arch.model_type:
            equals_token = tokenizer.vocab["="]
            model = construct_crammed_recurrent(cfg_arch, tokenizer.vocab_size, equals_token)

    if model is not None:  # Return local model arch
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with architecture {cfg_arch.model_type} loaded with {num_params:,} parameters.")
        return model

    try:  # else try on HF
        model = construct_huggingface_model(cfg_arch, tokenizer.vocab_size)
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with config {cfg_arch} loaded with {num_params:,} parameters.")
        return model
    except Exception as e:
        raise ValueError(f"Invalid model architecture {cfg_arch.model_type} given. Error: {e}")

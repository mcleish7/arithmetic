"""Instantiate backend objects in a congruent format."""
import torch

from .torch_default import initialize_torch

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)


def load_backend(model, tokenizer, cfg_train, cfg_impl, setup=_default_setup, init_compile_and_distribute=True):
    if cfg_impl.name == "torch-default":
        return initialize_torch(model, tokenizer, cfg_train, cfg_impl, setup=setup, init_compile_and_distribute=init_compile_and_distribute)
    else:
        raise ValueError(f"Invalid backend {cfg_impl.name} given.")

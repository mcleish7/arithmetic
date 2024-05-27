import logging
import os
import torch

import logging

from safetensors.torch import load_file, save_file
import cramming

log = logging.getLogger(__name__)


"""Utilities common to several backends."""
def group_parameters(model, cfg_train):
    model_parameters = list(model.named_parameters())
    if len(cfg_train.limited_decay_keys) > 0:
        grouped_parameters = optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model_parameters if not any(nd in n for nd in cfg_train.limited_decay_keys)],
                "weight_decay": cfg_train.optim.weight_decay,
            },
            {
                "params": [p for n, p in model_parameters if any(nd in n for nd in cfg_train.limited_decay_keys)],
                "weight_decay": 0.0,
            },
        ]
    else:
        grouped_parameters = [p for n, p in model_parameters]
    return grouped_parameters


def get_model_engine_tokenizer_dataloaders(cfg, setup, train_eval: bool = True):
    """This function gets the model, model engine (if needed), tokenizer, and data"""
    if train_eval:
        train_eval_cfg = cfg.train
    else:
        train_eval_cfg = cfg.eval

    tokenizer_model = None
    cfg_arch = cfg.arch  # if not loading from checkpoint, need architecture config
    checkpoint_path = None
    try:
        # attempt to load latest in case of preemption
        prev_checkpoint_path = os.path.join(cfg.model_dir, cfg.name, "checkpoints")
        tokenizer_model, cfg_arch, checkpoint_path = cramming.utils.find_pretrained_checkpoint(
            "latest",
            local_checkpoint_folder=str(prev_checkpoint_path),
            arch_modifications=train_eval_cfg.arch_modifications
        )
        log.info(f"Getting latest checkpoint at {prev_checkpoint_path}")

    except:
        # no previous checkpoint saved.  Checking separate model directory
        if train_eval_cfg.checkpoint is not None:
            try:
                tokenizer_model, cfg_arch, checkpoint_path = cramming.utils.find_pretrained_checkpoint(
                    train_eval_cfg.checkpoint,
                    local_checkpoint_folder=cfg.model_dir,
                    arch_modifications=train_eval_cfg.arch_modifications
                )
                log.info(f"Found checkpoint at {cfg.model_dir} or {train_eval_cfg.checkpoint}")
                # importantly, if checkpoint is found, we will use that model arch, modifications doesnt seem to work.
            except Exception as e:
                log.info(f"Unable to load checkpoint {train_eval_cfg.checkpoint} or in directory {cfg.model_dir}."
                         f"  Initializing model from scratch!")

    log.info(f"Loading Data")
    datasets, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl, cfg.data_dir)

    real_dataset_sample_length = len(datasets['train'][0]['input_ids']) # for arithmetic datasets

    if tokenizer_model is not None:
        # todo consider if we even need to return the tokenizer with the checkpoint (only HF?)
        tokenizer = tokenizer_model
    dataloaders = cramming.prepare_dataloaders(datasets, tokenizer, train_eval_cfg, cfg.impl)

    log.info(f"Constructing Model")
    model = cramming.construct_model(cfg_arch, tokenizer)

    metadata = {}

    if train_eval:
        # if in train mode, need engine
        fully_init_model_to_begin = False if checkpoint_path is not None else True
        model_engine = cramming.load_backend(
            model,
            tokenizer,
            cfg.train,
            cfg.impl,
            setup=setup,
            init_compile_and_distribute=fully_init_model_to_begin, # false if we are planning to load a checkpoint in later
        )

        if checkpoint_path is not None:
            # load checkpoint, engine handles loaded model
            metadata = model_engine.load_checkpoint(cfg_arch, checkpoint_path)
            for k, v in dataloaders.items():
                try:
                    # for dataloaders with epochs (RuntimeInfiniteDataLoader) set that epoch to start here
                    v.set_epoch(metadata.get("epoch", 0))
                except:
                    pass

        model_engine.train(train_eval_cfg.pretrain_in_train_mode)
        model_engine.current_seq_length = real_dataset_sample_length # setting the number of tokens seen correctly for arithmetic data
    else:
        if checkpoint_path is not None:
            model = load_model_checkpoint(model, checkpoint_path)
        model_engine = None
    return model, model_engine, tokenizer, dataloaders, metadata


def load_model_checkpoint(model, model_dir, forward_only_model_with_skip=False):
    ext = "model.safetensors"
    try:
        model_file = os.path.join(model_dir, ext)
        model_state = load_file(model_file)
    except:
        ext = "state_dict.pth"
        model_file = os.path.join(model_dir, ext)
        loaded = torch.load(model_file)
        model_state = loaded.get("model_state", None)

    if model_state is None:
        raise ValueError(f"No model found in directory {model_dir} (in '/state_dict.pth' or '/model.safetensors')")
    else:
        log.info(f"Loading Model from {model_file}")

    if "encoder.embedding.word_embedding.weight" not in model_state:
        # Hack to save space when saving the model, more clever though would be save the right one in the first place
        model_state["encoder.embedding.word_embedding.weight"] = model_state["decoder.weight"]
    sanitized_state = {}
    try:
        for k, v in model_state.items():
            if k.startswith("module."):
                k = k[7:]
            if forward_only_model_with_skip:
                if "_orig_mod" in k: # we load in original model to here so we can drop this
                    k = k.replace("._orig_mod", "")
            sanitized_state[k] = v

        model.load_state_dict(sanitized_state, strict=True)
        log.info("finished loading state dict")
    except RuntimeError as e:
        log.info(f"State dict difference is {str(e).split('Error(s) in loading state_dict for')[1]}... Ok?")
        exit()

    return model

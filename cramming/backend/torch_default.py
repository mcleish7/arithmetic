"""Basic training backend engine for pytorch training with all bells and whistles.

Interface set up to be compliant with the deepspeed engine interface.


There are two versions here, the TorchEngineMinimal, which is the default, and TorchEngineFull which contains a few training variations
that were tested but ultimately discarded, so read that part only if you're interested.

"""

import json
import logging
import os
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, Union

import torch
import torch._inductor.utils
import transformers
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers.utils.generic import working_or_temp_dir

from .optimizers import LARS, SAM, ProgressiveBatching
from .optimizers.schedulers import get_schedule_fn

# from .utils import group_parameters, prepare_pretraining_dataloader, prepare_validation_dataloader
from .utils import group_parameters, load_model_checkpoint

log = logging.getLogger(__name__)
_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)
import warnings
from ..utils import flatten

warnings.filterwarnings("ignore", "Detected call of ", UserWarning)  # schedulers are deliberately used differently


def initialize_torch(model, tokenizer, cfg_train, cfg_impl, setup=_default_setup, init_compile_and_distribute=True):
    """initialize a torch engine."""
    model_engine = TorchEngine(
        model,
        cfg_train,
        cfg_impl,
        setup=setup,
        seq_length=tokenizer.model_max_length,
        init_compile_and_distribute=init_compile_and_distribute,
    )
    model_engine.train()
    return model_engine


class TorchEngine(torch.nn.Module):
    """This class mirrors deepspeed functionality and hides variable batch sizes, microbatching, AMP details and compilation"""

    def __init__(self, model, cfg_train, cfg_impl, setup=_default_setup, seq_length=128, init_compile_and_distribute=True):
        """Load Engine. The model will be compiled by default.
        init_compile_and_distribute=False => In the case we are loading in a checkpoint we might aswell not send it across GPUs as this will be redone later
        """

        super().__init__()

        self.cfg_train = cfg_train
        self.cfg_impl = cfg_impl
        if self.cfg_impl.microbatch_size is None:
            self.cfg_impl.microbatch_size = self.cfg_train.batch_size
        if self.cfg_impl.microbatch_size > self.cfg_train.batch_size:
            raise ValueError(f"MBS is {self.cfg_impl.microbatch_size}, but BS is only {self.cfg_train.batch_size}.")
        self.current_seq_length = seq_length

        # Mixed Precision:
        enabled = self.cfg_impl.mixed_precision if setup["device"].type != "cpu" else False
        # Modules like LN are unsupported on CPU amp, so mixed precision args are disregarded on CPU
        # See https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior and check for layer_norm
        enable_scaling = self.cfg_impl.grad_scaling and self.cfg_impl.mixed_precision and setup["device"].type != "cpu"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaling)
        amp_dtype = getattr(torch, self.cfg_impl.mixed_precision_target_dtype) if setup["device"].type != "cpu" else torch.bfloat16
        self.amp_settings = dict(device_type=setup["device"].type, enabled=enabled, dtype=amp_dtype)

        # Choose setup and move model
        self.setup = setup
        model.to(**self.setup)
        self._original_model = model
        log.info("Compiling model, in the Constructor of TorchEngine")
        model = torch.compile(
            model,
            mode=self.cfg_impl.mode,
            dynamic=self.cfg_impl.dynamic,
            fullgraph=self.cfg_impl.fullgraph,
            backend=self.cfg_impl.backend,
            disable=not cfg_impl.compile_torch,
            # detailed options; cannot be given at the same time as mode:
            options=flatten(cfg_impl._inductor_vars, parent_key="", sep=".") if cfg_impl._inductor_vars is not None else None,
        )

        if torch.distributed.is_initialized():
            if init_compile_and_distribute:
                log.info("Distributing model, in the Constructor of TorchEngine")
                self.model = self._init_distributed(model)
            else:
                log.info(
                    "<WARNING> NOT Distirbuting model in the Constructor of TorchEngine, we will attempt to do this later as we are loading in a checkpoint"
                )
                self.model = model
            self.num_machines = torch.distributed.get_world_size()
        else:
            self.model = model
            self.model.no_sync = nullcontext
            self.num_machines = 1

        # Microbatch accumulation settings and counters
        self.effective_mbs = self.cfg_impl.microbatch_size * self.num_machines  # across machines
        self.current_batch_size = self.cfg_train.batch_size if self.cfg_train.batch_size_ramp == 0 else self.effective_mbs
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs
        self.accumulated_samples = 0  # Record the number of samples seen, reset after triggering gradient update
        self.steps = 0  # Record the number of times "step" has been triggered
        self.steps_since_reset = 0  # Record the number of times "step" has been triggered

        self.initial_time = time.time()
        self.previous_elapsed_time = 0.0
        self.optimizer, self.scheduler = _load_optimizer(model, cfg_train, cfg_impl, self.previous_elapsed_time, self.get_true_budget())

    def get_true_budget(self):
        return (
            min(self.cfg_train.budget, self.cfg_train.overall_budget - self.previous_elapsed_time / 3600)
            + self.previous_elapsed_time / 3600
        )

    def step(self, batch: dict[str, torch.Tensor]):
        loss = self.forward(**batch)["loss"]
        self.backward(loss)
        self.optimizer_step()
        return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids"]):
        """Move batch of data into device memory."""
        device_batch = {
            k: v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            for k, v in batch.items()
            if k in keys  # Add more keywords here if needed
        }
        return device_batch

    def forward(self, *inputs, **kwargs):
        self.accumulated_samples += self.effective_mbs
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
            with torch.autocast(**self.amp_settings):
                return self.model(*inputs, **kwargs)

    def backward(self, loss):
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
            return self.scaler.scale(loss / self.accumulation_steps_expected).backward()

    @torch.no_grad()
    @torch._dynamo.disable()
    def forward_inference(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            outputs = self.model(*inputs, **kwargs)["logits"]
        predictions = outputs.argmax(dim=-1)
        return outputs, predictions

    @torch._dynamo.disable()
    @torch.inference_mode()
    def dynamic_generation(self, *inputs, temperature=0.7, token_limit=100, **kwargs):
        with torch.autocast(**self.amp_settings):
            try:
                if hasattr(self._original_model, "_generate"):  # my signature
                    outputs = self._original_model._generate(*inputs, temperature=temperature, token_limit=token_limit, **kwargs)
                elif hasattr(self._original_model, "generate"):  # hf signature
                    outputs = self._original_model.generate(
                        *inputs, do_sample=True, num_beams=1, temperature=temperature, max_new_tokens=token_limit, **kwargs
                    )
                else:
                    raise NotImplementedError()
            except Exception as e:  # Fallback
                log.info(f"Falling back to default generation scheme due to error {e} in model._generate or model.generate.")
                # Generate new tokens the dumb way as a fall-back
                # need to implement the improved way for transformers eventually
                device_inputs = inputs[0]
                predicted_ids = []
                for gen_idx in range(token_limit):
                    logits = self._original_model(device_inputs, *inputs[1:], **kwargs)["logits"]
                    predicted_token = torch.multinomial(torch.softmax(logits * temperature, dim=-1), 1)
                    device_inputs = torch.cat([device_inputs, predicted_token], dim=-1)
                    predicted_ids += [predicted_token]
                outputs = torch.cat(predicted_ids, dim=-1)
        return outputs

    def optimizer_step(self):
        """Requires a scheduler that is based on iterations instead of epochs."""
        self.steps += 1
        self.steps_since_reset += 1
        if self.accumulated_samples >= self.current_batch_size:
            self.accumulated_samples = 0

            if self.cfg_train.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_train.gradient_clipping, norm_type=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.schedule_batch_size()
        self.scheduler.step()  # Trigger in every step, otherwise things get annoying with grad accumulation

    def set_train_batch_size(self, batch_size):
        """Allow dynamic modifications of batch size."""
        self.current_batch_size = batch_size
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs

    def schedule_batch_size(self):
        """Optionally implement linear batch size ramp-ups."""
        mbs = self.effective_mbs

        if (self.cfg_train.batch_size_ramp > 0) and (self.cfg_train.batch_size_ramp < 1):
            # interpret as percentage of total budget
            elapsed_time = (time.time() - self.initial_time) + self.previous_elapsed_time
            elapsed_hours = elapsed_time / 60 / 60
            fake_step = int(elapsed_hours / self.get_true_budget() * self.cfg_train.steps)
            # WARNING: this does not correctly pick up from checkpoint if elapsed>budget i.e. going over the orginal budget may cause a problem here

            batch_size_step = self.cfg_train.batch_size / (self.cfg_train.steps * self.cfg_train.batch_size_ramp)

            new_batch_size = min(int(fake_step * batch_size_step // mbs + 1) * mbs, self.cfg_train.batch_size)
        elif self.steps < self.cfg_train.batch_size_ramp:
            batch_size_step = self.cfg_train.batch_size / self.cfg_train.batch_size_ramp
            new_batch_size = int(self.steps * batch_size_step // mbs + 1) * mbs
        else:
            new_batch_size = self.cfg_train.batch_size
        self.set_train_batch_size(new_batch_size)

    def record_batch_size(self):
        if self.cfg_train.optim_mod.name != "progressive-batching":
            return self.current_batch_size
        else:
            return self.optimizer.last_full_step_accumulation * self.current_batch_size

    def record_tokens_per_step(self):
        """Tokens in each microbatch step."""
        return self.current_seq_length * self.effective_mbs

    @torch.no_grad()
    def retrieve_model_state_dict(self):
        if self.cfg_impl.compile_torch:
            if torch.distributed.is_initialized():
                state_dict = self.model.module._orig_mod.state_dict()  # ughhhh
            else:
                state_dict = self.model._orig_mod.state_dict()  # ugh
        else:
            if torch.distributed.is_initialized():
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        return state_dict

    def _init_distributed(self, model):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.setup["device"]] if self.setup["device"].type == "cuda" else None,
            output_device=self.setup["device"] if self.setup["device"].type == "cuda" else None,
            broadcast_buffers=self.cfg_impl.broadcast_buffers,
            bucket_cap_mb=self.cfg_impl.bucket_cap_mb,
            gradient_as_bucket_view=self.cfg_impl.gradient_as_bucket_view,
            static_graph=self.cfg_impl.static_graph,
        )
        return model

    def load_checkpoint(self, cfg_arch, file, skip_optim_state=False) -> Dict[str, Any]:
        """Load list of states from checkpoint file. Not generally compatible with any other engine?"""
        self.optimizer.zero_grad()
        # defaults
        metadata = {"epochs": 0, "steps": 0, "loss": 0, "data_idx": 0, "elapsed_time": 0.0}
        if file.startswith("hf://"):
            if file.endswith("-untrained"):
                log.info("Loading NO pretrained model as a sanity check ...")
            else:
                self.model = self.model.from_pretrained(file.split("hf://")[1], config=cfg_arch).to(**self.setup)
                # reinit optimizer:
                self.optimizer, self.scheduler = _load_optimizer(
                    self.model, self.cfg_train, self.cfg_impl, metadata.get("elapsed_time", 0.0), self.get_true_budget()
                )
        else:
            # we load back into original model as we want to redistribute the weights across ranks to be super sure!!
            model = load_model_checkpoint(self._original_model, file)
            model.to(**self.setup)
            # reinitialising the model as we are losing the compile speed otherwise
            model = torch.compile(
                model,
                mode=self.cfg_impl.mode,
                dynamic=self.cfg_impl.dynamic,
                fullgraph=self.cfg_impl.fullgraph,
                backend=self.cfg_impl.backend,
                disable=not self.cfg_impl.compile_torch,
                # detailed options; cannot be given at the same time as mode:
                options=flatten(self.cfg_impl._inductor_vars, parent_key="", sep=".") if self.cfg_impl._inductor_vars is not None else None,
            )
            if torch.distributed.is_initialized():
                self.model = self._init_distributed(model)
                log.info("Recompiled and distributed")
            else:
                self.model = model
                log.info("Recompiled")

            if not skip_optim_state:
                state_file = os.path.join(file, "state_dict.pth")
                try:
                    loaded = torch.load(state_file)
                    optim_state = loaded["optim_state"]
                    scheduler_state = loaded["scheduler_state"]
                    scaler_state = loaded["scaler_state"]
                    metadata = loaded["metadata"]
                    self.load_metadata(metadata)

                    # this is mainly so that the scheduler knows about the elapsed time
                    self.optimizer, self.scheduler = _load_optimizer(
                        self.model, self.cfg_train, self.cfg_impl, self.previous_elapsed_time, self.get_true_budget()
                    )
                    self.optimizer.load_state_dict(optim_state)
                    self.scheduler.load_state_dict(scheduler_state)

                    self.scaler.load_state_dict(scaler_state)
                    log.info(f"Successfully loaded state with metadata {metadata}")
                except Exception as e:
                    raise ValueError(f"Error loading optimizer and scheduler states from {state_file}. {e}")
        return metadata

    def load_metadata(self, metadata: Dict[str, Any]):
        self.steps = metadata.get("steps", 0)
        self.previous_elapsed_time = metadata.get("elapsed_time", 0.0)
        # add other state things here

    def save_training_checkpoint(self, checkpoint_directory: str, checkpoint_name: Union[str, float], metadata: Dict[str, Any]):
        """Path, identifier and additional client state. This checkpoint can be used to resume training.
        The default behavior is to save this checkpoint relative to the training working directory.
        """

        os.makedirs(checkpoint_directory, exist_ok=True)
        full_path = os.path.join(checkpoint_directory, checkpoint_name)

        optim_state = self.optimizer.state_dict()
        model_state = self.retrieve_model_state_dict()
        scheduler_state = self.scheduler.state_dict()
        scaler_state = self.scaler.state_dict()
        state_dict = {
            "metadata": metadata,
            "optim_state": optim_state,
            "scaler_state": scaler_state,
            "scheduler_state": scheduler_state,
        }
        safetensor_name = f"{full_path}_model_state.pth"
        save_file(model_state, safetensor_name)
        other_name = f"{full_path}_non_model.pth"
        torch.save(state_dict, other_name)

    def save_final_model(self, base_directory, identifier, tokenizer, cfg_arch, dryrun=False):
        """This checkpoint can be used for downstream tasks.
        The default behavior is to save this checkpoint to a checkpoints folder under base_directory/name/checkpoints"""
        try:
            identifier_str = f"{identifier:2.4f}"
        except ValueError:
            identifier_str = str(identifier)
        full_path = os.path.join(base_directory, "checkpoints", identifier_str)
        os.makedirs(full_path, exist_ok=True)
        # This saves tokenizer_config.json, tokenizer.json and special_tokens_map.json to this folder
        if not dryrun:

            # Save model.safetensors, model_config.json
            save_file(self.retrieve_model_state_dict(), os.path.join(full_path, "model.safetensors"))
            # legacy save: torch.save(self.retrieve_model_state_dict(), os.path.join(full_path, "model.pth"))
            with open(os.path.join(full_path, "model_config.json"), "w") as file:
                json.dump(OmegaConf.to_container(cfg_arch, resolve=True), file)

    def save_model(
        self,
        checkpoint_directory: str,
        checkpoint_name: Union[str, float],
        cfg_arch,
        metadata: Dict[str, Any],
        tokenizer=None,
        save_safe: bool = False,
    ):
        """This checkpoint can be used for downstream tasks.
        The default behavior is to save this checkpoint to a checkpoints folder under base_directory/name/checkpoints"""
        full_path = os.path.join(checkpoint_directory, checkpoint_name)
        os.makedirs(full_path, exist_ok=True)

        with open(os.path.join(full_path, "model_config.json"), "w") as file:
            json.dump(OmegaConf.to_container(cfg_arch, resolve=True), file)

        model_state = self.retrieve_model_state_dict()
        state_dict = {
            "model_state": model_state,
        }

        if save_safe:
            # this is like the final checkpoint, saves as safetensor but doesn't save state
            model_state = state_dict.pop("model_state")
            save_file(model_state, os.path.join(full_path, "model.safetensors"))

        if metadata is not None:
            optim_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            scaler_state = self.scaler.state_dict()
            state_dict["metadata"] = metadata
            state_dict["optim_state"] = optim_state
            state_dict["scheduler_state"] = scheduler_state
            state_dict["scaler_state"] = scaler_state
        if len(state_dict) > 0:
            # if save_safe this will only save non-model stuff
            state_dict_path = os.path.join(full_path, "state_dict.pth")
            torch.save(state_dict, state_dict_path)

        return full_path

    def push_to_hub(self, tokenizer, cfg, dryrun=False):
        """Analogous to save_final_model, but save model to hugginface hub."""
        from huggingface_hub import HfApi
        from io import BytesIO

        api = HfApi()

        if not dryrun:
            log.info(f"Pushing model to hub repository {cfg.impl.hf_directoy_name}.")
            final_state_dict = self.retrieve_model_state_dict()
            self.model.load_state_dict(final_state_dict)

            # Push model with safetensors:
            # This is a manual modification of model.push_to_hub which doesn't support safetensors yet
            repo_id = cfg.impl.hf_directoy_name
            if os.path.isdir(repo_id):
                working_dir = repo_id
                repo_id = repo_id.split(os.path.sep)[-1]
            else:
                working_dir = repo_id.split("/")[-1]
            repo_id = self.model._create_repo(repo_id)
            use_temp_dir = not os.path.isdir(working_dir)
            with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
                files_timestamps = self.model._get_files_timestamps(work_dir)
                # Save all files.
                self.model.save_pretrained(
                    work_dir,
                    max_shard_size="10GB",
                    safe_serialization=True,
                    state_dict=self.retrieve_model_state_dict(),
                )
                self.model._upload_modified_files(
                    work_dir,
                    repo_id,
                    files_timestamps,
                    commit_message=None,
                    token=None,
                    create_pr=None,
                )
            # Push tokenizer:
            tokenizer.push_to_hub(cfg.impl.hf_directoy_name)
            # Push config files:
            for config_group, config_name in zip([cfg.arch, cfg.data, cfg.train], ["arch", "data", "train"]):
                buffer = BytesIO()
                buffer.write(json.dumps(OmegaConf.to_container(config_group, resolve=True), indent=4).encode())
                api.upload_file(
                    path_or_fileobj=buffer,
                    path_in_repo=f"{config_name}_budget_hours_{cfg.budget}.json",
                    repo_id=f"{api.whoami()['name']}/{cfg.impl.hf_directoy_name}",
                    # there has to be a better way to do this, but ...
                    repo_type="model",
                )
        else:
            log.info(f"Skipping huggingface upload in dryrun state. Would upload to {cfg.impl.hf_directoy_name}.")


def _load_optimizer(model, cfg_train, cfg_impl, elapsed_time=0.0, true_budget=-1):
    # Filter some parameters
    grouped_parameters = group_parameters(model, cfg_train)

    # Select optimizer implementation
    if cfg_train.optim.type == "AdamW":
        optimizer_class = torch.optim.AdamW
    elif cfg_train.optim.type == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg_train.optim.type == "RAdam":
        optimizer_class = torch.optim.RAdam
    elif cfg_train.optim.type == "SGD":
        optimizer_class = torch.optim.SGD
    elif cfg_train.optim.type == "Adafactor":
        optimizer_class = transformers.Adafactor
    elif cfg_train.optim.type == "Shampoo":
        optimizer_class = Shampoo
    elif cfg_train.optim.type == "AdaHessian":
        optimizer_class = Adahessian
    elif cfg_train.optim.type == "AdamWScale":
        optimizer_class = AdamWScale
    elif cfg_train.optim.type == "Sophia-G":
        optimizer_class = Sophia
    elif cfg_train.optim.type == "Lion":
        from lion_pytorch import Lion

        optimizer_class = Lion

    elif cfg_train.optim.type == "Adam8bit":
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.Adam8bit
    elif cfg_train.optim.type == "AGD":
        depth = len(list(model.parameters()))
        optimizer_class = partial(AGD, depth=depth)
    else:
        raise ValueError(f"Invalid optimizer {cfg_train.optim.type} given.")
    optimizer_args = {k: v for k, v in cfg_train.optim.items() if k != "type"}
    if cfg_impl.foreach_optimizer and cfg_train.optim.type != "Shampoo":
        optimizer_args["foreach"] = True

    if torch.distributed.is_initialized() and cfg_impl.zero_redundancy_optimizer:
        # The overlap option is a whole bucket of problems in itself for now...
        optimizer = ZeroRedundancyOptimizer(
            grouped_parameters,
            optimizer_class=optimizer_class,
            parameters_as_bucket_view=True,
            overlap_with_ddp=False,
            **optimizer_args,
        )
    else:
        optimizer = optimizer_class(grouped_parameters, **optimizer_args)

    if cfg_train.optim_mod.name == "none":
        optimizer_to_schedule = optimizer
    else:
        optim_params = {k: v for k, v in cfg_train.optim_mod.items() if k != "name"}
        if cfg_train.optim_mod.name == "LARS":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "LARC":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "SAM":
            optimizer = SAM(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "progressive-batching":
            optimizer = ProgressiveBatching(optimizer, **optim_params)

        optimizer_to_schedule = optimizer.optim

    scheduler = get_schedule_fn(cfg_train, elapsed_time=elapsed_time, true_budget=true_budget)(optimizer_to_schedule)

    return optimizer, scheduler

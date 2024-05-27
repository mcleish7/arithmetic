"""Script for a pretraining run."""

import torch
import hydra

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming

log = logging.getLogger(__name__)


def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    model, model_engine, tokenizer, dataloaders, prior_metadata = cramming.backend.get_model_engine_tokenizer_dataloaders(
        cfg, setup, True)

    data_source = list(cfg.data.sources.values())[0]["provider"]
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time()
    last_save_time = wallclock_timer
    train_time = time.time()  # Crude time measurement for print_loss_every_nth_step
    training_allowed = True
    loss_vals, loss_ppls = [], []

    loss = prior_metadata.get("loss", 0)
    total_steps = prior_metadata.get("steps", 0)
    epochs = prior_metadata.get("epochs", 0)
    elapsed_time = prior_metadata.get("elapsed_time", 0.0)
    prev_data_idx = prior_metadata.get("data_idx", 0)

    # Launch training
    log.info(f"Training run for {cfg.budget} hours{f'' if cfg.overall_budget < 0 else f' and {cfg.overall_budget} hours overall'}{f'' if elapsed_time <= 0 else f' of which {elapsed_time/3600:.2f} hours was used so far.'}")
    run_time = min(cfg.budget, cfg.overall_budget - elapsed_time/3600)
    log.info(f"Running for {run_time:.2f} hours")
    if run_time <= 0:
        log.info(f"Already used budget!")
        return {}

    for data_idx, batch in enumerate(dataloaders["train"], prev_data_idx):
        logged_stats = False

        device_batch = model_engine.to_device(batch)
        model_outputs = {}
        for seq_idx in range(0, max(1, device_batch["input_ids"].shape[1] - cfg.train.stream_depth), cfg.train.stream_depth):
            # Run over seq_dim and dispatch multiple model updates while maintaining state in model_outputs
            # .clone() is required for new nightly so compilation is not stuck recompiling due to StorageOffsets
            input_ids = device_batch["input_ids"][:, seq_idx: seq_idx + cfg.train.stream_depth + 1].clone()  # last token is only a target
            model_outputs = model_engine.forward(input_ids=input_ids, **model_outputs)
            loss = model_outputs["loss"]

            model_engine.backward(loss)
            model_engine.optimizer_step()
            loss_vals.append(loss.detach())
            loss_ppls.append(model_outputs["log_perplexity"].detach())

            if cfg.dryrun:
                break

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget, elapsed_time, cfg.overall_budget) or data_idx == cfg.train.steps:
            training_allowed = False

            log.info(f"Reached deadline: Used {get_time_elapsed(wallclock_timer)/3600:.2f}/{cfg.budget} hours {'' if cfg.overall_budget < 0 else f' since reset and {get_time_elapsed(wallclock_timer, elapsed_time)/3600:.2f}/{cfg.overall_budget} hours overall'}. "
                     f"Stopping training ...")
                     
        if check_checkpointing(data_idx, cfg.impl, last_save_time):
            if cramming.utils.is_main_process():
                loss_vals, loss_ppls, train_time = collect_stats(
                    data_idx,
                    loss_vals,
                    loss_ppls,
                    model_outputs,
                    train_time,
                    stats,
                    model_engine,
                    dataloaders["train"],
                    cfg,
                )
                logged_stats = True

                # Save intermediate training checkpoint?
                epochs = dataloaders["train"].epoch_counter
                last_save_time = time.time()
                last_save_time_datetime = datetime.datetime.fromtimestamp(last_save_time)
                if cfg.impl.save_intermediate_model_name is None:
                    # if name is given use it (will overwrite), else use time to save
                    checkpoint_name = f"{cfg.arch.model_type}_{last_save_time_datetime.strftime('%Y-%m-%d')}_{loss.item():2.4f}"
                else:
                    checkpoint_name = cfg.impl.save_intermediate_model_name
                checkpoint_path = os.path.join(cfg.model_dir, cfg.name, "checkpoints")

                metadata = {"epochs": epochs,
                            "loss": loss.item(),
                            "data_idx": data_idx,
                            "steps": model_engine.steps,
                            "elapsed_time": (time.time() - wallclock_timer) + elapsed_time
                            }

                saved_path_temp = model_engine.save_model(checkpoint_path, checkpoint_name, cfg.arch, metadata)
                log.info(
                    f"Saving training checkpoint! Number of epochs/optim steps/data steps trained for: {epochs}/{model_engine.steps}/{data_idx},"
                    f"saving to: {saved_path_temp}")

                if cfg.impl.push_to_huggingface_hub:
                    model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)

        # Collect stats and print to console and upload to wandb
        if data_idx % cfg.impl.print_loss_every_nth_step == 0:
            if not logged_stats:
                loss_vals, loss_ppls, train_time = collect_stats(
                    data_idx,
                    loss_vals,
                    loss_ppls,
                    model_outputs,
                    train_time,
                    stats,
                    model_engine,
                    dataloaders["train"],
                    cfg,
                )

            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination, elapsed_time):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        if not loss.detach().isfinite():
            log.info(f"Non-finite loss in block {data_idx} on device {cfg.impl.local_rank}.")
            training_allowed = False

        flag_communication(training_allowed)

        if (cfg.dryrun and data_idx > (model_engine.accumulation_steps_expected + 1)) or not training_allowed:
            break

    epochs = dataloaders["train"].epoch_counter
    log.info(f"Number of epochs/optim steps/data steps trained for: {epochs}/{model_engine.steps}/{data_idx}")

    if cramming.utils.is_main_process():
        # Save final checkpoint?
        if cfg.impl.save_final_model:
            metadata = {"epochs": epochs,
                        "loss": loss.item(),
                        "data_idx": data_idx,
                        "steps": model_engine.steps,
                        "elapsed_time": time.time() - wallclock_timer + elapsed_time
                        }
                        
            if cfg.model_dir is None:
                save_dir = cfg.base_dir
            else:
                save_dir = cfg.model_dir
            checkpoint_path = os.path.join(save_dir, cfg.name, "checkpoints")
            checkpoint_name = f"FINAL_{loss.item():2.4f}"
            saved_path = model_engine.save_model(checkpoint_path, checkpoint_name, cfg.arch, metadata, None, save_safe=True)

            log.info(f"Saving training checkpoint to: {saved_path}")

            if cfg.impl.push_to_huggingface_hub:
                model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)
            
            # Print some example completions
        if loss.detach().isfinite():
            generate(model_engine, tokenizer, cfg.impl.example_prompts, token_limit=cfg.impl.example_token_limit)
    
    # Save to summary:
    if loss.detach().isfinite():
        validation_log_p = validate(model_engine, dataloaders["test"], setup, cfg)
    else:
        validation_log_p = float("Inf")
    log.info(f"Log-Perplexity on validation data is {validation_log_p:2.4f}.")
    metrics = dict(
        validation_log_ppl=validation_log_p,
        validation_ppl=torch.as_tensor(validation_log_p).exp().item(),
        num_params=sum([p.numel() for p in model.parameters()]),
    )

    return metrics


def get_time_elapsed(start_time: float, additional_time: float = 0.0) -> float:
    return time.time() - start_time + additional_time

def check_checkpointing(data_idx: int, cfg_impl, last_save_time) -> bool:
    step_condition = cfg_impl.save_every_nth_step > 0 and (data_idx % cfg_impl.save_every_nth_step == 0)
    time_condition = cfg_impl.save_every_n_minutes > 0 and (time.time() - last_save_time) / 60 > cfg_impl.save_every_n_minutes
    return cfg_impl.save_intermediate_checkpoints and (step_condition or time_condition)


def check_deadline(launch_time, hour_limit, prev_budget: float = 0.0, overall_hour_limit: float = 0.0):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    overall_budget = overall_hour_limit if overall_hour_limit >= 0 else hour_limit
    current_violated = (current_time - launch_time) / 3600 > hour_limit
    overall_violated = (prev_budget + (current_time - launch_time)) / 3600 > overall_budget
    return current_violated or overall_violated


def check_early_termination(start_time, loss, early_termination, prev_budget: float = 0.0):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        overall_budget = early_termination.overall_budget if early_termination.overall_budget > 0 else early_termination.budget
        current_violated = (current_time - start_time) / 3600 > early_termination.budget
        overall_violated = (prev_budget + (current_time - start_time)) / 3600 > overall_budget
        return current_violated or overall_violated
    else:
        return False


def collect_stats(data_step, loss_vals, log_ppls, model_outputs, train_time, stats, model_engine, dataloader, cfg):
    """ "data_step" here refers to one step on the dataloader, which may be multiple steps on the model_engine."""
    stats["data_step"] += [data_step]
    stats["epoch"] += [dataloader.epoch_counter]
    stats["model_steps"] += [model_engine.steps]

    tokens_per_step = model_engine.record_tokens_per_step()
    stats["tokens"] += [data_step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss
    stats["log_ppl"] += [torch.stack(log_ppls).mean().item()]  # Averaged loss
    if "losses" in model_outputs:
        for key, acccum_loss in model_outputs["losses"].items():
            if key != "count":
                stats[key] += [acccum_loss.item()]
    if "logits" in model_outputs:
        try:
            precise_logits = model_outputs["logits"].to(dtype=torch.float32)
            stats["entropy"] += [torch.distributions.Categorical(torch.softmax(precise_logits, dim=-1)).entropy().mean().item()]
        except ValueError:
            stats["entropy"] += [float("NaN")]  # can happen if invalid values in logits, or softmax numerical issues

    current_lr = model_engine.optimizer.param_groups[0].get("lr", float("NaN"))
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at data block {data_step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if data_step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per block ({tokens_per_second:.0f}t/s). "
        # log_msg += f"Est.for all sched. blocks: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals, log_ppls = [], []
    train_time = time.time()
    return loss_vals, log_ppls, train_time


@torch.no_grad()
def validate(model_engine, validloader, setup, cfg):
    """Evaluate on validation set."""
    log.info("Starting model validation.")
    model_engine.eval()
    val_timer = time.time()
    # Cut up smaller streams so the inductor doesn't break, but keep parallelizable archs at full depth:
    eval_depth = 1 if cfg.train.stream_depth < cfg.data.seq_length else cfg.data.seq_length

    log_perplexity = 0
    len_validloader = len(validloader)

    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch)
        seq_len = max(1, device_batch["input_ids"].shape[1] - eval_depth)
        num_entries = len(range(0, seq_len))
        # Stream over sequence
        model_outputs = {}
        for seq_idx in range(0, seq_len, eval_depth):
            input_ids = device_batch["input_ids"][:, seq_idx : seq_idx + eval_depth + 1].clone()  # last token is used as target
            model_outputs = model_engine.forward(input_ids=input_ids, **model_outputs)
            log_perplexity += model_outputs.get("log_perplexity", model_outputs["loss"].detach()) / num_entries
            if cfg.dryrun:
                break

        if step % cfg.impl.print_loss_every_nth_step == 0:
            log_msg = f"Avg Log-Perplexity: {log_perplexity/(step + 1):2.4f} at step {step} "
            if step > 1:
                validation_time = (time.time() - val_timer) / cfg.impl.print_loss_every_nth_step
                estimated_train_finish = str(datetime.timedelta(seconds=validation_time * len(validloader)))
                tokens_per_step = cramming.utils.num_processes() * model_engine.record_tokens_per_step()
                tokens_per_second = tokens_per_step / validation_time
                log_msg += f" Perf: {validation_time:2.4f}s per step ({tokens_per_second:.0f}t/s). "
                log_msg += f"Estimated Total validation Time: {estimated_train_finish}."

            val_timer = time.time()
            log.info(log_msg)
        
        if step > 200000: # putting hard limit of 200,000 steps for validation
            len_validloader = step
            break

        if cfg.dryrun:
            break

    model_engine.train(cfg.train.pretrain_in_train_mode)
    return log_perplexity.item() / len_validloader


def generate(model_engine, tokenizer, example_prompts, token_limit=10, temp=1.0):
    model_engine.eval()
    # Just do a dumb generation for now, can implement efficient generation later
    for prompt in example_prompts:

        tokenized_inputs = torch.as_tensor(tokenizer(prompt)["input_ids"], dtype=torch.long)[None, :]#-1]  # cut off EOT NOT ALWAYS SAFE
        print("tokenised input is ",tokenized_inputs)
        device_inputs = model_engine.to_device(dict(input_ids=tokenized_inputs))["input_ids"]
        print("device inputs: ", device_inputs)
        # Generate new tokens
        predicted_ids = model_engine.dynamic_generation(device_inputs, temperature=temp, token_limit=token_limit)
        print("predicted ids: ", predicted_ids, " with length ", predicted_ids.shape)
        # print(type(predicted_ids[0]))
        decoded_completion = tokenizer.decode(predicted_ids[0].tolist())  # drop batch dim before decoding

        log.info(f"[{prompt}] {decoded_completion}")


def flag_communication(training_allowed):
    """A quick and dirty communication through NCCL. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor = torch.as_tensor(training_allowed).cuda()
        torch.distributed.all_reduce(comm_tensor, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor >= 1:
            return True
        else:
            return False
    else:
        return training_allowed


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.3")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")


if __name__ == "__main__":
    launch()

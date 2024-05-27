import logging
import hydra
from omegaconf import OmegaConf
import cramming
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import re
import pandas as pd
import datasets
import os
from typing import List, Dict
from cramming.data.tokenizer_preparation import get_tokenizer
import random

log = logging.getLogger(__name__)

def grid_plotter(data, type="accs", name='_large', extra_path=None):
    """plot a 2d accuracy grid"""
    data = np.array(data)*100
    df = pd.DataFrame(data)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", annot_kws={'size': 8,'rotation':0})
    
    # Customize the plot
    plt.title("Accuracy - percetange, rounded to 1dp")
    plt.ylabel("1st Number Length")
    plt.xlabel("2nd Number Length")
    size = data.shape[0]
    plt.xticks(np.arange(0.5, size+0.5, 1), labels=np.arange(1, size+1, 1))
    plt.yticks(np.arange(0.5, size+0.5, 1), labels=np.arange(1, size+1, 1))

    if extra_path is not None:
        plt.savefig(f"{extra_path}{type}{name}_grid_plot", bbox_inches='tight')
    else:
        plt.savefig(f"{type}{name}_grid_plot", bbox_inches='tight')
    plt.clf()

def grid_logic(cfg):
    """logic to select function to control which part of a 2d grid this run should be responsible for evaling"""

    # origional testing
    def logic_func_large(data_size_1, data_size_2):
        return (data_size_1 <= 23 or data_size_2 <=23)
    logic_func = logic_func_large
    name = '_large'
    max_size = 23+1
    
    if cfg.ood_only:
        def logic_func_ood(data_size_1, data_size_2):
            return (data_size_1 >=24 or data_size_2 >=24) and (data_size_1 <= 30 or data_size_2 <=30)
        logic_func = logic_func_ood
        name = '_ood_only'
        max_size = 30+1
        
    if cfg.up_to_40:
        def logic_func_40(data_size_1, data_size_2):
            return (data_size_1 >=31 or data_size_2 >=31) and (data_size_1 <=40 or data_size_2 <=40)
        logic_func = logic_func_40
        name = '_up_to_40'
        max_size = 40+1
        
    if cfg.up_to_50:
        def logic_func_50(data_size_1, data_size_2):
            return (data_size_1 >=41 or data_size_2 >=41) and (data_size_1 <=50 or data_size_2 <=50)
        logic_func = logic_func_50
        name = '_up_to_50'
        max_size = 50+1

    # checkerboarding: for the large eval we can checkerboard:

    if cfg.checkerboard is not None:
        if cfg.checkerboard == 'even':
            def checkerboard_even(data_size_1, data_size_2):
                return ((data_size_1+data_size_2)%2 ==0)
            checkerboard_func = checkerboard_even
            checkerboard_str = "_even"
        elif cfg.checkerboard == 'odd':
            def checkerboard_odd(data_size_1, data_size_2):
                return ((data_size_1+data_size_2)%2 ==1)
            checkerboard_func = checkerboard_odd
            checkerboard_str = "_odd"
        else:
            print("checkerboard config not allowed")
            exit()
    else:
        def always_true(data_size_1, data_size_2):
            return True
        checkerboard_func = always_true
        checkerboard_str = ""


    # if we are testing up to 100, split into 10 steps each of approximately equal number of forward passes required
    if cfg.big_eval_step_1: # 1 -> 46
        def logic_func_big_1(data_size_1, data_size_2):
            return (data_size_1 <= 46 and data_size_2 <= 46) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_1
        name = '_big_eval_1'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_2: # 47 -> 58
        def logic_func_big_2(data_size_1, data_size_2):
            return (data_size_1 >=47 or data_size_2 >=47) and (data_size_1 <=58 and data_size_2 <=58) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_2
        name = '_big_eval_2'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_3: # 59 -> 67
        def logic_func_big_3(data_size_1, data_size_2):
            return (data_size_1 >=59 or data_size_2 >=59) and (data_size_1 <=67 and data_size_2 <=67) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_3
        name = '_big_eval_3'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_4: # 68 -> 74
        def logic_func_big_4(data_size_1, data_size_2):
            return (data_size_1 >=68 or data_size_2 >=68) and (data_size_1 <=74 and data_size_2 <=74) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_4
        name = '_big_eval_4'+checkerboard_str
        max_size = 100+1
      
    if cfg.big_eval_step_5: # 75 -> 80
        def logic_func_big_5(data_size_1, data_size_2):
            return (data_size_1 >= 75 or data_size_2 >=75) and (data_size_1 <=80 and data_size_2 <=80) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_5
        name = '_big_eval_5'+checkerboard_str
        max_size = 100+1

    if cfg.big_eval_step_6: # 81 -> 85
        def logic_func_big_6(data_size_1, data_size_2):
            return (data_size_1 >= 81 or data_size_2 >=81) and (data_size_1 <=85 and data_size_2 <=85) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_6
        name = '_big_eval_6'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_7: # 86 -> 90
        def logic_func_big_7(data_size_1, data_size_2):
            return (data_size_1 >= 86 or data_size_2 >=86) and (data_size_1 <=90 and data_size_2 <=90) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_7
        name = '_big_eval_7'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_8: # 91 -> 94
        def logic_func_big_8(data_size_1, data_size_2):
            return (data_size_1 >= 91 or data_size_2 >=91) and (data_size_1 <=94 and data_size_2 <=94) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_8
        name = '_big_eval_8'+checkerboard_str
        max_size = 100+1
    
    if cfg.big_eval_step_9: # 95 -> 97
        def logic_func_big_9(data_size_1, data_size_2):
            return (data_size_1 >= 95 or data_size_2 >=95) and (data_size_1 <=97 and data_size_2 <=97) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_9
        name = '_big_eval_9'+checkerboard_str
        max_size = 100+1
        
    if cfg.big_eval_step_10: # 98 -> 100
        def logic_func_big_10(data_size_1, data_size_2):
            return (data_size_1 >= 98 or data_size_2 >=98) and (data_size_1 <=100 and data_size_2 <=100) and checkerboard_func(data_size_1, data_size_2)
        logic_func = logic_func_big_10
        name = '_big_eval_10'+checkerboard_str
        max_size = 100+1

    # boolean_list_precidence = [large, ood_only, up_to_40, up_to_50, big_eval_step_1, big_eval_step_2, big_eval_step_3, big_eval_step_4, big_eval_step_5]

    log.info(f"large = {cfg.large}")
    log.info(f"ood only = {cfg.ood_only}")
    log.info(f"up to 40 = {cfg.up_to_40}")
    log.info(f"up to 50 = {cfg.up_to_50}")
    log.info(f"big eval 1 = {cfg.big_eval_step_1}")
    log.info(f"big eval 2 = {cfg.big_eval_step_2}")
    log.info(f"big eval 3 = {cfg.big_eval_step_3}")
    log.info(f"big eval 4 = {cfg.big_eval_step_4}")
    log.info(f"big eval 5 = {cfg.big_eval_step_5}")
    log.info(f"big eval 6 = {cfg.big_eval_step_6}")
    log.info(f"big eval 7 = {cfg.big_eval_step_7}")
    log.info(f"big eval 8 = {cfg.big_eval_step_8}")
    log.info(f"big eval 9 = {cfg.big_eval_step_9}")
    log.info(f"big eval 10 = {cfg.big_eval_step_10}")
    log.info(f"the last true value in the above list will be run, mul and pos arith can take control after this")

    return logic_func, name, max_size

def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.name, "checkpoints")
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg.eval.checkpoint,
                                                                                local_checkpoint_folder,
                                                                                cfg.eval.arch_modifications)
    if cfg.max_rec is not None: # can have more/less recurrences for eval
        cfg_arch.maximal_recurrence_in_eval = cfg.max_rec
    else:
        cfg_arch.maximal_recurrence_in_eval = cfg_arch.maximal_recurrence
    log.info(f"cfg_arch.maximal_recurrence_in_eval changed to {cfg_arch.maximal_recurrence_in_eval}")
    cfg_arch.throttle = False # turn throttle off

    logic_func, name, max_size = grid_logic(cfg)

    # import tokeniser
    cfg_data_sources_values_list = list(cfg.data.sources.values())[0]
    if cfg_data_sources_values_list["provider"] == "arithmetic":
        tokenizer = get_tokenizer(cfg_data_sources_values_list["tokenizer_type"])
    else: 
        log.info("exiting as this is only for arithmetic")
        exit()
    vocab = tokenizer.ids_to_tokens
    EOS_token = tokenizer._convert_token_to_id(tokenizer.eos_token)
    PAD_token = tokenizer._convert_token_to_id(tokenizer.pad_token)
    assert PAD_token == 0, "PAD token must be token zero for our code to work"

    # Load model
    if 'alpha' not in cfg_arch:
        cfg_arch['alpha'] = 1.0

    model = cramming.construct_model(cfg_arch, tokenizer).to(device)
    model = cramming.backend.load_model_checkpoint(model, model_file)
    model.to(device)
    model.eval()

    log.info(f"greedy = {cfg.greedy}, note: if greedy = True this overrides any temperature arguments")
    ## Greedy decoding will overide any temperature arguments

    if cfg.max_size_given is not None: # allows unique splits for eval
        max_size = cfg.max_size_given

    # Grid plots - grid search from 1x1 to 12x12 data
    data_sizes = list(range(1, max_size))
    acc_grid = np.zeros((len(data_sizes),len(data_sizes)))
    start_ind_1 = 0
    start_ind_2 = 0
    tuple_method = False
    completed_one = False
    if "big_eval" in name:
        tuple_method = True
        # go up two layers and search for grid
        try:
            with open(f"../../accs_grid_quick{name}.json", 'r') as file:
                data = json.load(file)
            start_ind_1 = data[1]
            start_ind_2 = data[2]
            acc_grid = np.array(data[0])
            log.info("loaded grid from previous run")
        except:
            pass

    if cfg.start_ind_1_given is not None: # allows unique splits for eval
        start_ind_1 = cfg.start_ind_1_given
    if cfg.start_ind_2_given is not None:
        start_ind_2 = cfg.start_ind_2_given
    log.info(f"start_ind_1 = {start_ind_1}, start_ind_2 = {start_ind_2}")

    os.makedirs("outputs", exist_ok=True)

    all_outputs_folder_path = f"../../all_outputs_max_recurrence={cfg_arch.maximal_recurrence_in_eval}"
    os.makedirs(all_outputs_folder_path, exist_ok=True)

    if not cfg.extended_eval:
        # main 2d loop
        for data_size_1 in data_sizes:
            for data_size_2 in data_sizes:
                proceed = False
                if data_size_1 >= start_ind_1 or data_size_2 >= start_ind_2:
                    proceed = True

                if not proceed:
                    continue

                # check if done
                # if done it will be done and saved in f"../../acc_for_{data_size_1}_{data_size_2}.txt"
                if os.path.exists(f"{all_outputs_folder_path}/acc_for_{data_size_1}_{data_size_2}.txt"):
                    with open(f"{all_outputs_folder_path}/acc_for_{data_size_1}_{data_size_2}.txt", 'r') as file:
                        acc = float(file.read())
                    acc_grid[data_size_1-1, data_size_2-1] = acc
                    continue

                if logic_func(data_size_1, data_size_2):
                    completed_one = True
                    log.info(f"Starting iteration in grid eval for size: {data_size_1} and {data_size_2}")
                    # only one option -- sorting with reversed numbers
                    file_path = f"../../../../data/arithmetic_data/sort_reverse/sort_uniform_distribution_sort_basic_max_digits_n_{data_size_1}_max_length_m_{data_size_2}_200_p_00_reverse_all/hf_tokenized_dataset"
                   
                    tokenized_dataset = datasets.load_from_disk(file_path)["test"]
                    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=100, shuffle=False)

                    # keep track of totals for a batch as we only eval one sample at a time
                    correct_total = 0
                    all_total = 0
                    top_1_total = 0
                    for batch in data_loader:
                        input_ids = batch["input_ids"]
                        input_ids = torch.stack(input_ids).to(device)
                        input_ids = torch.transpose(input_ids, 0, 1)

                        all = 0
                        correct = 0
                        top_1 = 0
                        for i in range(len(input_ids)):
                            example = input_ids[i]
                            equals_token = tokenizer._convert_token_to_id("=")
                            equals_indices = torch.where(example == equals_token)[0].item()
                            question = example[:equals_indices + 1]
                            answer = example[equals_indices + 1:]
                            
                            question = question.unsqueeze(0)

                            local_token_limit = int(len(answer) * 2)
                            predicted_ids = model._generate(question,
                                                            token_limit=local_token_limit,
                                                            temperature=cfg.temp,
                                                            steps_at_generation_time=cfg_arch.maximal_recurrence_in_eval,
                                                            greedy=cfg.greedy, quick=True)
                            predicted_ids = predicted_ids.squeeze()

                            # get the answer
                            eos_token = tokenizer._convert_token_to_id(tokenizer.eos_token)
                            eos_indices = torch.where(answer == eos_token)[0].item()
                            answer = answer[:eos_indices]

                            predicted_ids = predicted_ids[:len(answer)]
                            if torch.equal(predicted_ids, answer):
                                correct += 1

                            top_1_target = answer[0]
                            top_1_predicted = predicted_ids[0]
                            if torch.equal(top_1_target, top_1_predicted):
                                top_1 += 1

                            all += 1

                        correct_total += correct
                        top_1_total += top_1
                        all_total += all


                    acc = correct_total / all_total
                    acc_top_1 = top_1_total / all_total

                    log.info(f"accuracy for data that has numbers "
                             f"with maximum number of digits as {data_size_1} , "
                             f"and the array of length {data_size_2} is {acc * 100}")
                    log.info(f"Top 1 accuracy for data that has numbers "
                             f"with maximum number of digits as {data_size_1} , "
                             f"and the array of length {data_size_2} is {acc_top_1 * 100}")

                    question = tokenizer.decode(question.squeeze().tolist())
                    answer = tokenizer.decode(answer.tolist())
                    predicted = tokenizer.decode(predicted_ids.tolist())
                    log.info(f"For example : sort {question} for which the answer is {answer} , "
                             f"and the predicted is {predicted}")
                    acc_grid[(data_size_1-1), (data_size_2-1)] = acc * 100
                    
                    # save all in case of crash
                    with open(f"{all_outputs_folder_path}/acc_for_{data_size_1}_{data_size_2}.txt", "w") as file:
                        file.write(f"{acc * 100}")
                    with open(f"{all_outputs_folder_path}/top_1_acc_for_{data_size_1}_{data_size_2}.txt", "w") as file:
                        file.write(f"{acc_top_1 * 100}")

        log.info(f"acc grid: {acc_grid}")

        with open(f"accs_grid_quick_{start_ind_1}_{start_ind_2}_{max_size}.json", "w") as file:
            json.dump(acc_grid.tolist(), file)

        # Grid plots - one for accs one for contains
        grid_plotter(acc_grid, name=f"{start_ind_1}_{start_ind_2}_{max_size}")
        grid_plotter(acc_grid, name=f"{start_ind_1}_{start_ind_2}_{max_size}", extra_path=all_outputs_folder_path)

    log.info("Eval complete")

@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.3")
def launch(cfg):
    log.info("calling main launch")
    cfg = cramming.utils.pathfinder(cfg)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    main(cfg)

if __name__ == "__main__":
    launch()
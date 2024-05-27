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
    sns.heatmap(df, cmap="YlGnBu", fmt=".1f", annot_kws={'size': 8,'rotation':0})
    
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

def index_hints_helper(num, tokenizer):
    """Add index hints into a tokenized number"""
    char_set = tokenizer.char_set
    shape1 = num.shape[1]
    for i in range(shape1):
        this_char_token = tokenizer._convert_token_to_id(char_set[i])
        char_to_insert = this_char_token * torch.ones((num.shape[0], 1), dtype=num.dtype, device=num.device)
        num = torch.cat((num[:,:(2*i)], char_to_insert, num[:,(2*i):]), dim=1)
    return num

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

    if cfg.mul: # multiplication
        def logic_func_for_mul(data_size_1, data_size_2):
            return (data_size_1 <= 25 or data_size_2 <= 25)
        logic_func = logic_func_for_mul
        name = '_large'
        max_size = 25+1
    log.info(f"mul = {cfg.mul}")

    if cfg.pos_arth: # bitwise OR
        def logic_func_for_pos(data_size_1, data_size_2):
            return (data_size_1 <= 25 or data_size_2 <= 25)
        logic_func = logic_func_for_pos
        name = '_large'
        max_size = 25+1
    log.info(f"pos_arth = {cfg.pos_arth}")

    if cfg.pos_arth_ood:
        def logic_func_for_pos_ood(data_size_1, data_size_2):
            return (data_size_1 >= 26 or data_size_2 >=26) and (data_size_1 <=40 and data_size_2 <=40)
        logic_func = logic_func_for_pos_ood
        name = '_ood_only'
        max_size = 40+1
    log.info(f"pos_arth_ood = {cfg.pos_arth_ood}")

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
        max_size = max_size_given

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

    if not cfg.extended_eval:
        # main 2d loop
        for data_size_1 in data_sizes:
            for data_size_2 in data_sizes:
                if (data_size_1 < start_ind_1 or data_size_2 < start_ind_2) and not completed_one:
                    continue
                else:
                    proceed = False
                    # if both data sizes are less than the start indices, then dont proceed
                    # but if one of them is greater than the start indices, then proceed
                    if data_size_1 >= start_ind_1 or data_size_2 >= start_ind_2:
                        proceed = True
                        
                    if not proceed:
                        continue

                print(f"evaluating for {data_size_1} and {data_size_2}")

                if logic_func(data_size_1, data_size_2):
                    completed_one = True
                    log.info(f"Starting iteration in grid eval for size: {data_size_1} and {data_size_2}")
                    correct_total = 0

                    # get the correct dataset, these names may need to be changed if you make new datasets
                    file_path = f"../../../../data/arithmetic_data/+_grid_eval_dataset_padded_tokenized/+_n_{data_size_1}_m_{data_size_2}_examples_100_diff_lens_seed_42/hf_tokenized_dataset"
                    if cfg.reverse_inputs:
                        file_path = f"../../../../data/arithmetic_data/+_grid_eval_dataset_reverse_all_tokenized/+_n_{data_size_1}_m_{data_size_2}_examples_100_diff_lens_seed_42/hf_tokenized_dataset"
                    if cfg.mul:
                        file_path = f"../../../../data/arithmetic_data/x_grid_eval_dataset_2_reverse_all_tokenized/x_n_{data_size_1}_m_{data_size_2}_examples_100_diff_lens_exact_seed_91/hf_tokenized_dataset"
                    if cfg.pos_arth or cfg.pos_arth_ood:
                        file_path = f"../../../../data/arithmetic_data/pos_or_one_vec_zeros_eval/or_one_vec_zeros_{data_size_1}_{data_size_2}/hf_tokenized_dataset"
                    tokenized_dataset = datasets.load_from_disk(file_path)["test"]
                    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=100, shuffle=False)
                    equals_tensor = data_size_1+data_size_2+6
                    if cfg.pos_arth or cfg.pos_arth_ood:
                        equals_tensor = data_size_1+data_size_2+2

                    for batch in data_loader:
                        # split prompt and answer
                        tokenized_prompts = batch["input_ids"][:equals_tensor]
                        tokenized_prompts = torch.stack(tokenized_prompts).to(device)
                        tokenized_prompts = torch.transpose(tokenized_prompts, 0, 1)
                        tokenized_answers = batch["input_ids"][equals_tensor:]
                        tokenized_answers = torch.stack(tokenized_answers).to(device)
                        tokenized_answers = torch.transpose(tokenized_answers, 0, 1)
   
                        if cfg.remove_padding and (cfg_data_sources_values_list["tokenizer_type"] != "index"):
                            # removes the padding from the eval data
                            num1 = tokenized_prompts[:,:data_size_1]
                            op = tokenized_prompts[:,data_size_1+1:data_size_1+2]
                            num2 = tokenized_prompts[:,data_size_1+3:data_size_1+data_size_2+3]
                            equals = tokenized_prompts[:,data_size_1+data_size_2+4:data_size_1+data_size_2+5]
                            tokenized_prompts = torch.cat((num1, op, num2, equals), dim=1)
 
                        if cfg_data_sources_values_list["tokenizer_type"] == "index":
                            # adding in the index hints to the input numbers
                            num1 = tokenized_prompts[:,:data_size_1]
                            num1 = index_hints_helper(num1, tokenizer)
                            op = tokenized_prompts[:,data_size_1+1:data_size_1+2]
                            num2 = tokenized_prompts[:,data_size_1+3:data_size_1+data_size_2+3]
                            num2 = index_hints_helper(num2, tokenizer)
                            equals = tokenized_prompts[:,data_size_1+data_size_2+4:data_size_1+data_size_2+5]
                            tokenized_prompts = torch.cat((num1, op, num2, equals), dim=1)

                            predicted_ids = None

                            ## below inserts the characters for the model, we decided against this in the end
                            predicted_ids = model._generate(tokenized_prompts, token_limit=(tokenized_answers.shape[1]*2), temperature=cfg.temp, steps_at_generation_time=cfg_arch.maximal_recurrence_in_eval, greedy=cfg.greedy, quick=True)
                            predicted_ids = torch.transpose(predicted_ids, 0, 1)

                            new_tensor = torch.zeros_like(predicted_ids)
                            for i in range(predicted_ids.size(0)): # inefficient!!
                                # Filter out values greater than 17
                                filtered_values = predicted_ids[i][predicted_ids[i] <= 17]
                                # Place filtered values in new tensor and pad with zeros
                                new_tensor[i, :len(filtered_values)] = filtered_values

                            predicted_ids = new_tensor[:, :tokenized_answers.shape[1]] # trim off the excess
                            predicted_ids = torch.transpose(predicted_ids, 0, 1)

                        else:
                            predicted_ids = model._generate(tokenized_prompts, token_limit=tokenized_answers.shape[1], temperature=cfg.temp, steps_at_generation_time=cfg_arch.maximal_recurrence_in_eval, greedy=cfg.greedy, quick=True)
                        
                        if len(predicted_ids.shape) > 1: # i.e. we have a batch of more than one
                            predicted_ids = torch.transpose(predicted_ids, 0, 1)
                        else:
                            predicted_ids = predicted_ids.reshape((1,-1)) # add a batch dim otherwise
                            
                    # ignore everything after EOS on eval but replacing all after EOS with PAD
                    eval_tensor = predicted_ids.clone()
                    input_tensor_EOS = (eval_tensor == EOS_token).int()
                    indices_of_EOS = torch.argmax(input_tensor_EOS, dim=1)
                    mask = torch.arange(eval_tensor.size(1)).to(device) > indices_of_EOS[:, None]
                    eval_tensor[mask] = PAD_token
                    
                    # compare eval tensor to correct outputs
                    elementwise_equal = torch.eq(eval_tensor, tokenized_answers)
                    rows_equal = torch.all(elementwise_equal, dim=1)
                    num_equal_rows = torch.sum(rows_equal).item()
                    correct_total += (num_equal_rows/tokenized_prompts.shape[0])
                    log.info(f"accuracy for {data_size_1}, {data_size_2}: {num_equal_rows} = {correct_total*100}%")

                    # combine the prompts and outputs
                    complete_lines = torch.cat((tokenized_prompts,predicted_ids), dim=1)
                    tokens_list = complete_lines.tolist()
                    decoded_batch = list(map(lambda seq: list(map(lambda token: vocab[token], seq)), tokens_list)) # map token ids to tokens
                    log.info(f"example for {data_size_1}, {data_size_2}: {decoded_batch[0]}")
                    # save the answers down so we don't eval twice ever
                    with open(f"outputs/+_n_{data_size_1}_m_{data_size_2}.json", 'w') as json_file:
                        json.dump(decoded_batch, json_file)

                    acc_grid[(data_size_1-1),(data_size_2-1)] = correct_total

                    if tuple_method:
                        with open(f"../../accs_grid_quick{name}.json", "w") as file:
                            tuple_to_save = (acc_grid.tolist(),data_size_1,data_size_2)
                            json.dump(tuple_to_save, file)

        log.info(f"acc grid: {acc_grid}")

        with open(f"accs_grid_quick{name}.json", "w") as file:
            json.dump(acc_grid.tolist(), file)
        
        # Grid plots - one for accs one for contains
        grid_plotter(acc_grid, name=name)

    if cfg.extended_eval:
        # extended eval to eval large numbers easily, used the large eval numebers to split up into multiple parts

        number = int(re.findall(r'\d+', name)[0])
        log.info("starting extended eval")
        # this is hard coded for reverse all, addition past 100x100 grid, removing the padding

        accs = dict()
        batch_size_extended_eval = 100

        old_data_path = None
        for root, dirs, files in os.walk("../.."):
            if f"over_100_{number}.json" in files:
                old_data_path = os.path.join(root, f"over_100_{number}.json")

        if number == 1:
            start = 101
            list_to_do = range(start,161)
        elif number == 2:
            list_to_do = [1000, 800]
        elif number == 3:
            list_to_do = [200, 700, 900]
        elif number == 4:
            list_to_do = [300, 400, 500, 600]
        else:
            print("number too high")
            exit()

        if old_data_path is not None: # read the old accs dict and don't repeat what we have already done
            with open(old_data_path, 'r') as file:
                data = json.load(file)
            accs = {int(k): v for k, v in data.items()}
            to_do = set(list_to_do).difference(set(accs.keys()))
            list_to_do = list(to_do)

        log.info(f"In extended eval with number {number}")

        for data_size in list_to_do:
            log.info(f"Extended eval {data_size}")
            correct_total = 0
            file_path = f"../../../../data/arithmetic_data/+_grid_eval_dataset_reverse_all_tokenized_over_100/+_n_{data_size}_m_{data_size}_examples_100_diff_lens_exact_seed_42/hf_tokenized_dataset"
            tokenized_dataset = datasets.load_from_disk(file_path)["test"]
            data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size_extended_eval, shuffle=False)
            equals_tensor = data_size+data_size+6

            for batch in data_loader:
                # get prompt and answer
                tokenized_prompts = batch["input_ids"][:equals_tensor]
                tokenized_prompts = torch.stack(tokenized_prompts).to(device)
                tokenized_prompts = torch.transpose(tokenized_prompts, 0, 1)
                tokenized_answers = batch["input_ids"][equals_tensor:]
                tokenized_answers = torch.stack(tokenized_answers).to(device)
                tokenized_answers = torch.transpose(tokenized_answers, 0, 1)

                # remove the padding
                num1 = tokenized_prompts[:,:data_size]
                op = tokenized_prompts[:,data_size+1:data_size+2]
                num2 = tokenized_prompts[:,data_size+3:data_size+data_size+3]
                equals = tokenized_prompts[:,data_size+data_size+4:data_size+data_size+5]
                tokenized_prompts = torch.cat((num1, op, num2, equals), dim=1)

                # get the output from the model
                predicted_ids = model._generate(tokenized_prompts, token_limit=tokenized_answers.shape[1], temperature=cfg.temp, steps_at_generation_time=cfg_arch.maximal_recurrence_in_eval, greedy=cfg.greedy, quick=True)
                predicted_ids = torch.transpose(predicted_ids, 0, 1) # add a batch dim

                eval_tensor = predicted_ids.clone()
                input_tensor_EOS = (eval_tensor == EOS_token).int()
                indices_of_EOS = torch.argmax(input_tensor_EOS, dim=1)
                mask = torch.arange(eval_tensor.size(1)).to(device) > indices_of_EOS[:, None]
                eval_tensor[mask] = PAD_token
                elementwise_equal = torch.eq(eval_tensor, tokenized_answers)
                
                rows_equal = torch.all(elementwise_equal, dim=1)
                num_equal_rows = torch.sum(rows_equal).item()
                correct_total += (num_equal_rows/tokenized_prompts.shape[0])
                log.info(f"accuracy for {data_size}, {data_size}: {num_equal_rows} = {correct_total*100}%")

                # combine the prompts and outputs
                complete_lines = torch.cat((tokenized_prompts,predicted_ids), dim=1)
                tokens_list = complete_lines.tolist()
                decoded_batch = list(map(lambda seq: list(map(lambda token: vocab[token], seq)), tokens_list)) # map token ids to tokens
                log.info(f"example for {data_size}, {data_size}: {decoded_batch[0]}")
                # save the answers down so we don't eval twice ever

            accs[data_size] = correct_total
            with open(f"over_100_{number}.json", 'w') as json_file:
                    json.dump(accs, json_file)
                    
    log.info("Eval complete")

@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.3")
def launch(cfg):
    log.info("calling main launch")
    cfg = cramming.utils.pathfinder(cfg)
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    main(cfg)

if __name__ == "__main__":
    launch()
## combine multiple testing plots and make a pretty one 

import os
import numpy as np
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
import glob
import re

def grid_plotter(data, type="accs", path="", title=None, rect_size=20):
    """Plot the large 100x100 grid"""
    if title is None:
        title = "All numbers are percetanges rounded to 1dp"
    data = np.array(data)*100
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    annotate = False
    # use interpolant
    sns.heatmap(df, annot=annotate, cmap="YlGnBu", fmt=".0f", annot_kws={'size': 8,'rotation':0})

    rect = patches.Rectangle((0, 0), rect_size, rect_size, linewidth=1.8, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    rect_size = data.shape[0]
    plt.xticks(np.arange(1, rect_size+1, 2) - 0.5, labels=np.arange(1, rect_size+1, 2), rotation=90, fontsize=10)
    plt.yticks(np.arange(1, rect_size+1, 2) - 0.5, labels=np.arange(1, rect_size+1, 2), rotation=0, fontsize=10)
    
    # Customize the plot
    plt.title(title)
    plt.ylabel("1st Number Length")
    plt.xlabel("2nd Number Length")
    
    plt.savefig(f"{path}combined_accs_grid_plot_big_run", bbox_inches='tight', dpi=300)
    plt.clf()

def main():
    # replace with your model name
    model_name = "cramming-data/add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_abacus_attn_emb_nope_with_skip_connections_run_1"
    rect_size = 20

    directory_path = f"{model_name}/downstream"
    # get latest checkpoint for the model data
    config_path = f"{model_name}/checkpoints"
    all_checkpoints = [f for f in os.listdir(config_path)]
    checkpoint_paths = [os.path.join(config_path, c) for c in all_checkpoints]
    checkpoint_name = max(checkpoint_paths, key=os.path.getmtime)
    with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))
    max_rec = cfg_arch['maximal_recurrence']
    layers_in_block = cfg_arch['layers_in_recurrent_block']
    mask_bf_eq = cfg_arch['mask_before_equals']
    attn_type = cfg_arch['attention']['type']
    loss_reduc = cfg_arch['loss_reduction']
    throttle = cfg_arch['throttle']
    title = f"Model name:\n{model_name[14:]}\nNum layers in block: {layers_in_block}, Num blocks in training: {max_rec}\n Mask all before equals: {mask_bf_eq}, Train time: 24 hr\n attn: {attn_type}, temp: Greedy{', loss: 'if loss_reduc == 'none' else ''}{', throttle' if throttle else ''}"


    # Define the pattern to search for
    file_pattern = directory_path + "/accs_grid_quick_big_eval_?_even.json"
    matching_files_even = glob.glob(file_pattern, recursive=True)
    file_pattern = directory_path + "/accs_grid_quick_big_eval_??_even.json"
    matching_files_even += glob.glob(file_pattern, recursive=True)

    file_pattern = directory_path + "/accs_grid_quick_big_eval_?_odd.json"
    matching_files_odd = glob.glob(file_pattern, recursive=True)
    file_pattern = directory_path + "/accs_grid_quick_big_eval_??_odd.json"
    matching_files_odd += glob.glob(file_pattern, recursive=True)

    # Print the matching files
    number_pattern_even = re.compile(r'accs_grid_quick_big_eval_(\d+)_even.json')
    number_pattern_odd = re.compile(r'accs_grid_quick_big_eval_(\d+)_odd.json')

    # Print the matching files and the numbers extracted from them
    file_paths = []
    even_nums = []
    odd_nums = []

    for file_path in matching_files_even:
        match = number_pattern_even.search(file_path)
        if match:
            number = match.group(1)
            if number not in even_nums:
                even_nums.append(number)
                print("Number:", number)
            else:
                continue
        print("File:", file_path)
        file_paths.append(file_path)

    for file_path in matching_files_odd:
        match = number_pattern_odd.search(file_path)
        if match:
            number = match.group(1)
            if number not in odd_nums:
                odd_nums.append(number)
                print("Number:", number)
            else:
                continue
        print("File:", file_path)
        file_paths.append(file_path)

    arr = np.zeros((100, 100))
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if len(data) == 3:
                data = data[0]
        arr = arr + np.array(data)
        
    title = title + "\n Even: "+', '.join(sorted(even_nums, key=lambda x: int(x))) + "\n Odd: "+', '.join(sorted(odd_nums, key=lambda x: int(x)))
    grid_plotter(arr, type=type, path=f"{directory_path}/", title=title, rect_size=rect_size)
    print(f"{model_name}")

if __name__ == "__main__":
    main()
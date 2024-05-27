## combine multiple testing plots and make a pretty one 

import os
import numpy as np
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

def find_file(starting_directory, target_file):
    """Find target_file in the tree from starting_directory"""
    for root, dirs, files in os.walk(starting_directory):
        if target_file in files:
            return os.path.join(root, target_file)

def grid_plotter(data, type="accs", path="", title=None, rect_size=20, up_to_50=False):
    """plot the 2d grid (up to 50x50)"""
    if title is None:
        title = "All numbers are percetanges rounded to 1dp"
    data = np.array(data)*100
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".0f", annot_kws={'size': 8,'rotation':0})
    if up_to_50:
        rect = patches.Rectangle((0, 0), rect_size, rect_size, linewidth=1.5, edgecolor='red', facecolor='none')
    else:
        rect = patches.Rectangle((0, 0), rect_size, rect_size, linewidth=1, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    rect_size = data.shape[0]
    plt.xticks(np.arange(1, rect_size+1) - 0.5, labels=np.arange(1, rect_size+1), rotation=90, fontsize=10)
    plt.yticks(np.arange(1, rect_size+1) - 0.5, labels=np.arange(1, rect_size+1), rotation=0, fontsize=10)
    
    # Customize the plot
    plt.title(title)
    plt.ylabel("1st Number Length")
    plt.xlabel("2nd Number Length")
    
    plt.savefig(f"{path}combined_{type}_grid_plot{'_50' if up_to_50 else ''}", bbox_inches='tight', dpi=300)
    plt.clf()

def main():
    # replace with model name
    model_name = "cramming-data/add_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_abacus_attn_emb_nope_run_1"

    file_path = f"{model_name}/downstream"
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

    # works up in tiers starting from the smallest grid (large) up to the largest for this size (up_to_50)
    large_path = find_file(file_path, f"accs_grid_quick_large.json")
    with open(large_path, 'r') as file:
        data = json.load(file)
    large_data = np.array(data)

    ood_path = find_file(file_path, f"accs_grid_quick_ood_only.json")
    with open(ood_path, 'r') as file:
        data = json.load(file)
    ood_data = np.array(data)

    num_rows_to_add = ood_data.shape[0] - large_data.shape[0]
    num_cols_to_add = ood_data.shape[1] - large_data.shape[1]

    padded_array = np.pad(large_data, ((0, num_rows_to_add), (0, num_cols_to_add)), mode='constant', constant_values=0)
    combined = padded_array+ood_data

    rect_size=20
    path_40 = find_file(file_path, f"accs_grid_quick_up_to_40.json")
    if path_40 is not None:
        with open(path_40, 'r') as file:
            data = json.load(file)
        data_40 = np.array(data)
        num_rows_to_add = data_40.shape[0] - combined.shape[0]
        num_cols_to_add = data_40.shape[1] - combined.shape[1]
        padded_array = np.pad(combined, ((0, num_rows_to_add), (0, num_cols_to_add)), mode='constant', constant_values=0)
        combined = padded_array+data_40

    path_50 = find_file(file_path, f"accs_grid_quick_up_to_50.json")
    up_to_50 = False
    if path_50 is not None:
        with open(path_50, 'r') as file:
            data = json.load(file)
        data_50 = np.array(data)
        num_rows_to_add = data_50.shape[0] - combined.shape[0]
        num_cols_to_add = data_50.shape[1] - combined.shape[1]
        padded_array = np.pad(combined, ((0, num_rows_to_add), (0, num_cols_to_add)), mode='constant', constant_values=0)
        combined = padded_array+data_50
        up_to_50 = True
        
    grid_plotter(combined, type="accs", path=f"{file_path}/", title=title, rect_size=rect_size, up_to_50=up_to_50)

if __name__ == "__main__":
    main()
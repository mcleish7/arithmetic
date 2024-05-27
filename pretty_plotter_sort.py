import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def grid_plotter(data, title="", path=None):
    data = np.array(data)
    df = pd.DataFrame(data)

    # find the average accuracy
    avg = np.mean(data)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".1f", annot_kws={'size': 8, 'rotation': 0}, vmin=0, vmax=100)

    # Customize the plot
    plt.title(f"Accuracy - percetange, rounded to 1dp : {title}, Avg acc: {avg}")
    plt.ylabel("Maximum n-digit number (1-n)")
    plt.xlabel("Length of array to sort")
    size = data.shape[0]
    plt.xticks(np.arange(0.5, size + 0.5, 1), labels=np.arange(1, size + 1, 1))
    plt.yticks(np.arange(0.5, size + 0.5, 1), labels=np.arange(1, size + 1, 1))

    plt.savefig(f"{path}", bbox_inches='tight')
    plt.clf()


def run(names, short_hand, base_dir, sort_plots_path):
    os.makedirs(sort_plots_path, exist_ok=True)
    all_data_acc_dict = {}
    all_data_top_1_acc_dict = {}

    for i in range(len(names)):
        name = names[i]
        extra_name = short_hand[i]
        dict_key = extra_name[0]
        extra_name = extra_name[0] + "_" + extra_name[1]
        all_data_path = base_dir + name + "/downstream/"

        # get all the directories in the path that start with all_outputs
        all_dirs = os.listdir(all_data_path)
        # remove the ones that are not directories
        all_dirs = [dir for dir in all_dirs if os.path.isdir(all_data_path + dir)]
        all_images = []
        for dir in all_dirs:
            if "all_outputs" in dir:
                # get the recurrence
                recurrence = dir.split("_")[-1]
                if "recurrence" not in recurrence:
                    continue

                # get all the files in the directory
                files = os.listdir(all_data_path + dir + "/")
                all_images_local = []

                all_data_acc = {}
                all_data_top_1_acc = {}
                max_size = 0

                print(extra_name)
                print("dir", dir)

                for file in files:
                    if ".txt" in file:
                        all_info = file.split(".")[0]
                        all_info = all_info.split("_")
                        data_size_1 = int(all_info[-2])
                        data_size_2 = int(all_info[-1])

                        if data_size_1 > max_size:
                            max_size = data_size_1
                        if data_size_2 > max_size:
                            max_size = data_size_2

                        # get the accuracy
                        with open(all_data_path + dir + "/" + file, "r") as f:
                            acc = float(f.read())
                            if "top_1_acc" in file:
                                all_data_top_1_acc[(data_size_1, data_size_2)] = acc
                            else:
                                all_data_acc[(data_size_1, data_size_2)] = acc

                # create the grid plot
                data = np.zeros((max_size, max_size))
                for key in all_data_acc.keys():
                    data[key[0] - 1][key[1] - 1] = all_data_acc[key]
                grid_plotter(data,
                            title=f"{extra_name} {recurrence} acc",
                            path=f"./{sort_plots_path}/{extra_name}_{recurrence}_acc.png")

                if dict_key not in all_data_acc_dict.keys():
                    all_data_acc_dict[dict_key] = []
                    all_data_top_1_acc_dict[dict_key] = []

                all_data_acc_dict[dict_key].append(data)

                data = np.zeros((max_size, max_size))
                for key in all_data_top_1_acc.keys():
                    data[key[0] - 1][key[1] - 1] = all_data_top_1_acc[key]
                grid_plotter(data,
                            title=f"{extra_name} {recurrence} top_1_acc",
                            path=f"./{sort_plots_path}/{extra_name}_{recurrence}_top_1_acc.png")

                all_data_top_1_acc_dict[dict_key].append(data)


                all_images_local.append(cv2.imread(f"./{sort_plots_path}/{extra_name}_{recurrence}_acc.png"))
                all_images_local.append(cv2.imread(f"./{sort_plots_path}/{extra_name}_{recurrence}_top_1_acc.png"))
                all_images_local = cv2.hconcat(all_images_local)
                # write this image
                all_images.append((all_images_local, f"{extra_name}_{recurrence}.png"))

        os.makedirs(f"./{sort_plots_path}/final/", exist_ok=True)
        if len(all_images) == 1:
            all_images_local, name = all_images[0]
            cv2.imwrite(f"./{sort_plots_path}/final/{name}", all_images_local)
        else:
            os.makedirs(f"./{sort_plots_path}/final/{extra_name}/", exist_ok=True)
            for all_images_local, name in all_images:
                cv2.imwrite(f"./{sort_plots_path}/final/{extra_name}/{name}", all_images_local)

if __name__ == "__main__":
    names = ["sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_with_fire_8x1_1_24_run_1"]
    short_hand = [("rev_abacus_fire_8x1", "v1")] # the shrothand names for the runs you want to plot in the same order

    base_dir = "cramming-data/"
    sort_plots_path = "./sort_plots/"
    run(names, short_hand, base_dir, sort_plots_path)
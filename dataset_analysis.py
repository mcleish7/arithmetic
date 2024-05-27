import os
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import argparse

def read_dataset(dir_name, condense_white_space=False):
    # open all data files and append to big list
    dataset = []
    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_name, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                stripped_lines = [line.replace("\n", "") for line in lines]
                if condense_white_space:
                    stripped_lines = [re.sub('\s+',' ', line) for line in lines]
                dataset.extend(stripped_lines)

    for i in range(0,min(len(dataset),5)):
        print(dataset[i])
    return dataset

def remove_leading_zeros(match):
    """Removes all leading zeros"""
    return str(int(match.group(0)))

def count_digits(dataset, remove_formatting=False):
    """Count the digits in each operand"""
    pairs = {}
    input_1 = {}
    input_2 = {}
    ans = {}
    for input_string in dataset:
        cleaned_string = input_string.replace(' ', '')
        if remove_formatting:
            cleaned_string = re.sub(r'\b0+\d+', remove_leading_zeros, cleaned_string)

        numbers = re.findall(r'\d+', cleaned_string)
        digit_counts = [len(number) for number in numbers]

        input_1[digit_counts[0]] = input_1.get(digit_counts[0], 0) + 1
        input_2[digit_counts[1]] = input_2.get(digit_counts[1], 0) + 1
        ans[digit_counts[2]] = ans.get(digit_counts[2], 0) + 1

        input_tuple = (digit_counts[0], digit_counts[1])
        pairs[input_tuple] = pairs.get(input_tuple, 0) + 1

    return pairs, input_1, input_2, ans

def plot_pairs_heatmap(pairs, dir_name=".", remove_formatting=False):
    """plot a heatmap of the lengths of the operands"""
    max_length = int(max(max(pair) for pair in pairs.keys()))
    heatmap_matrix = np.zeros((max_length + 1, max_length + 1))

    # Populate the matrix with counts
    for pair, count in pairs.items():
        heatmap_matrix[pair[0],pair[1]] = count

    df = pd.DataFrame.from_dict(heatmap_matrix)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4g", cbar_kws={'label': 'Count'}, annot_kws={'size': 8,'rotation':45})
    plt.xlabel('Length of First Number')
    plt.ylabel('Length of Second Number')
    plt.title('Input Pairs Length Heatmap')
    plt.savefig(f"{dir_name}/pairs_heatmap{'_removed_prepended_zeros' if remove_formatting else ''}.png", bbox_inches='tight')
    plt.clf()

def line_plotter(data, name, dir_name=".", remove_formatting=False):
    """plot a line graph for the length of the operand """
    data = dict(sorted(data.items()))
    x_values = list(data.keys())
    y_values = list(data.values())

    # Plotting the line plot
    plt.plot(x_values, y_values, marker='o')

    # Adding labels and title
    plt.xlabel('Length of number')
    plt.ylabel('Count')
    plt.title(f"Line Plot for {name}")
    plt.savefig(f"{dir_name}/{name}_line_plot{'_removed_prepended_zeros' if remove_formatting else ''}.png", bbox_inches='tight')
    plt.clf()

def consecutive_digit_counts(input_strings):
    """Count the number of times a digit is repeated"""
    counts_by_digit = {}

    for input_str in input_strings:
        current_digit = None
        consecutive_count = 0

        for char in input_str:
            if char.isdigit():
                if char == current_digit:
                    consecutive_count += 1
                else:
                    if current_digit is not None:
                        # Update the dictionary with consecutive count
                        if consecutive_count != 1:
                            counts_by_digit.setdefault(current_digit, {}).setdefault(consecutive_count, 0)
                            counts_by_digit[current_digit][consecutive_count] += 1

                    current_digit = char
                    consecutive_count = 1

        # Update the dictionary for the last digit in the string
        if current_digit is not None:
            if consecutive_count != 1:
                counts_by_digit.setdefault(current_digit, {}).setdefault(consecutive_count, 0)
                counts_by_digit[current_digit][consecutive_count] += 1

    return counts_by_digit

def create_repetition_heatmap(data, dir_name=".", remove_formatting=False):
    """plot heat map for, consecutive_digit_counts"""
    data = dict(sorted(data.items()))
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0)

    # Create a heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".4g", cbar_kws={'label': 'Count'}, annot_kws={'size': 8,'rotation':45})
    plt.title('Consecutive Digit Counts Heatmap')
    plt.xlabel('Consecutive Count')
    plt.ylabel('Digit')
    plt.savefig(f"{dir_name}/repetition_count_heatmap{'_removed_prepended_zeros' if remove_formatting else ''}.png", bbox_inches='tight')
    plt.clf()

def main(dir_name):
    base_directory = "./cramming-data/data/arithmetic_data"
    dir_name = os.path.join(base_directory, dir_name)
    dataset = read_dataset(dir_name)

    options = [True, False]
    for remove_formatting in options:
        pairs, input_1, input_2, ans = count_digits(dataset, remove_formatting=remove_formatting)
        print(f"{'removed prepended zeros' if remove_formatting else 'keeping prepended zeros'}")
        print("pairs: ",pairs)
        print("input 1: ",input_1)
        print("input 2: ",input_2)
        print("answers: ",ans)

        plot_pairs_heatmap(pairs, dir_name=dir_name, remove_formatting=remove_formatting)
        line_plotter(input_1, "input_1", dir_name=dir_name, remove_formatting=remove_formatting)
        line_plotter(input_2, "input_2", dir_name=dir_name, remove_formatting=remove_formatting)
        line_plotter(ans, "answer", dir_name=dir_name, remove_formatting=remove_formatting)

        result_list = consecutive_digit_counts(dataset)
        print("repetitions: ",result_list)
        create_repetition_heatmap(result_list, dir_name=dir_name, remove_formatting=remove_formatting)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data analysis")
    parser.add_argument("--dir_name", type=str, required=True)
    FLAGS = parser.parse_args()

    main(FLAGS.dir_name)
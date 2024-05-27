import numpy as np
import argparse
import random
import os

def one_hot_vector(length, index=None):
    """return a one hot vector"""
    if index is None:
        index = np.random.randint(length)
    one_hot = np.zeros(length)
    one_hot[index] = 1
    return one_hot

def zero_vector(length):
    """return a zero vector"""
    zeros = np.zeros(length)
    return zeros

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--dir_name", type=str, required=True, help="dir to save to")
    parser.add_argument("--op", type=str, default='+', help="operation")
    parser.add_argument("--n", default=2, type=int, help="num digits in first number")
    parser.add_argument("--m", default=2, type=int, help="num digits in second number")
    parser.add_argument('--p', default=0.0, type=float, help="prob for adding padding")
    parser.add_argument("--max", default=-1, type=int, help="num digits in second number")
    parser.add_argument('--exact', action='store_true', help='only this size')
    parser.add_argument('--eval', action='store_true', help='save as part of eval dataset')
    FLAGS = parser.parse_args()

    p = FLAGS.p
    dir_name = FLAGS.dir_name
    lengths_n = lengths_n_range = list(range(1,FLAGS.n+1))
    lengths_m = lengths_m_range = list(range(1,FLAGS.m+1))
    if FLAGS.exact:
        lengths_n = [FLAGS.n]
        lengths_m = [FLAGS.m]
        
    ds = []
    # 2d loop to sample exaustively
    for i in lengths_n:
        for j in lengths_m:
            i_len=i
            j_len=j
            combined_len=max(i,j)
            for index in list(range(0,min(i,j))):
                if i_len > j_len: # put one hot in longer vector
                    vec1 = zero_vector(i_len)
                    vec2 = one_hot_vector(j_len, index)
                elif i_len < j_len:
                    vec1 = one_hot_vector(i_len, index)
                    vec2 = zero_vector(j_len)
                else: # i.e. same length so either can be the zeros
                    if random.random() > 0.5:
                        vec1 = one_hot_vector(i_len, index)
                        vec2 = zero_vector(j_len)
                    else:
                        vec1 = zero_vector(i_len)
                        vec2 = one_hot_vector(j_len, index)
                ans = one_hot_vector(combined_len, index)

                vec1_str = "".join(map(lambda x: str(int(x)), vec1))
                vec2_str = "".join(map(lambda x: str(int(x)), vec2))
                ans_str = "".join(map(lambda x: str(int(x)), ans))

                dataset_entry = f"{vec1_str}{FLAGS.op}{vec2_str}={ans_str}"
                
                if p>0: # add random padding, exponentially decaying
                    spaced_string = ""
                    for char in dataset_entry:
                        space_p = p
                        while random.random() < space_p:
                            space_p *= 0.1
                            spaced_string += " "
                        spaced_string += char
                    dataset_entry = spaced_string
            
                ds.append(dataset_entry)

    if FLAGS.max != -1:
        ds = random.sample(ds, min(len(ds),FLAGS.max)) # cut to maximum size
    if FLAGS.eval:
        data_dir = f"./cramming-data/data/arithmetic_data/pos_or_one_vec_zeros/{dir_name}"
        file_name = f"positional_arithmetic_n_{FLAGS.n}_m_{FLAGS.m}.txt"
    else:
        data_dir = f"./cramming-data/data/arithmetic_data/{dir_name}"
        file_name = f"positional_or_one_vec_zeros_n_{FLAGS.n}_m_{FLAGS.m}_examples_{len(ds)}.txt"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)

    with open(file_path, 'w') as file:
        for entry in ds:
            file.write(entry + '\n')
    print(f"created: {file_path}")

if __name__ == "__main__":
    main()

# input your model name and base_dir
name = "sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_reycle_with_fire_8x1_1_24_run_1"
base_dir = "cramming-data"

# pick which eval you are doing
add_100 = False
add_110+ = False
add_small = False
mul = False
sort = True
bitwise_or = False

# set the model parameters for eval
print("remember to edit max_rec and tokenizer!!")
max_rec = 1
tokenizer = ' data.sources.arithmetic.tokenizer_type="pad"'
if sort:
    tokenizer = ' data.sources.arithmetic.tokenizer_type="sort"'

## print statements for all tasks below
if add_100:
    for checkerboard_str in [" checkerboard=odd"," checkerboard=even"]:
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=55 big_eval_step_1=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=60 big_eval_step_2=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=70 big_eval_step_3=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=85 big_eval_step_4=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=90 big_eval_step_5=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=100 big_eval_step_6=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=100 big_eval_step_7=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=110 big_eval_step_8=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=110 big_eval_step_9=True reverse_inputs=True{tokenizer}{checkerboard_str}")
        print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=110 big_eval_step_10=True reverse_inputs=True{tokenizer}{checkerboard_str}")

if add_100:
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=105 big_eval_step_1=True reverse_inputs=True checkerboard=even extended_eval=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=105 big_eval_step_2=True reverse_inputs=True checkerboard=even extended_eval=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=105 big_eval_step_3=True reverse_inputs=True checkerboard=even extended_eval=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=105 big_eval_step_4=True reverse_inputs=True checkerboard=even extended_eval=True{tokenizer}")

if add_small:
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=30 reverse_inputs=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=35 ood_only=True reverse_inputs=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=45 up_to_40=True reverse_inputs=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=55 up_to_50=True reverse_inputs=True{tokenizer}")

if mul:
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=30 pos_arth=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=50 pos_arth_ood=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=30 mul=True{tokenizer}")

if sort:
    for i in range(0,30):
        print(f"python sort_eval.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} sort_reverse=True data.sources.arithmetic.tokenizer_type='sort' max_size_given={i+2} start_ind_1_given={i+1} start_ind_2_given={i+1}")

if bitwise_or: # we give data to evaluate up to 100x100 as we show in the paper, but the evaluation loop in only arithmetic_eval_quicker.py evaluates up to 40x40. This can be easily edited if required
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=30 pos_arth=True{tokenizer}")
    print(f"python arithmetic_eval_quicker.py name={name} base_dir={base_dir} data=arithmetic max_rec={max_rec} token_limit=50 pos_arth_ood=True{tokenizer}")
                    


# there is an automated helper in gen_eval_script.py for generating these evaluation scripts

# Addition
python arithmetic_eval_quicker.py name=<name> base_dir=$cramming_base_dir data=arithmetic max_rec=<max_rec> token_limit=105 big_eval_step_<STEP_NUM>=True reverse_inputs=True checkerboard=<EVEN/ODD> remove_padding=True data.sources.arithmetic.tokenizer_type="pad"

# Extended Addition Eval, i.e. 100
python arithmetic_eval_quicker.py name=<name> base_dir=$cramming_base_dir data=arithmetic max_rec=<max_Rec> token_limit=105 big_eval_step_5=True reverse_inputs=True checkerboard=even remove_padding=True extended_eval=True data.sources.arithmetic.tokenizer_type="pad"

# Multiplication
python arithmetic_eval_quicker.py name=<NAME> base_dir=$cramming_base_dir data=arithmetic max_rec=<max_rec> token_limit=30 mul=True data.sources.arithmetic.tokenizer_type="pad"

# Sorting
# max_size_given = end of grid, start_ind_... = start of grid, i.e. this evaluates from 1,1 to final_size, final_size
python sort_eval.py name=<name> base_dir=$cramming_base_dir data=arithmetic max_rec=<max_rec> sort_reverse=True data.sources.arithmetic.tokenizer_type='sort' max_size_given={final_size + 1} start_ind_1_given={1} start_ind_2_given={1}

# Bitwise OR
python arithmetic_eval_quicker.py name=<name> base_dir=$cramming_base_dir data=arithmetic max_rec=<max_rec> token_limit=105 big_eval_step_<STEP_NUM>=True checkerboard=<EVEN/ODD> pos_arth_ood=True data.sources.arithmetic.tokenizer_type="pad" remove_padding=False
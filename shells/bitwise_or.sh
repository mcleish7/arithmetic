# bitwise or is sometimes refered to as pos_arth in the code

## LT
# NOPE
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_1_16_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_nope_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=16 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None
#  FIRE
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_1_16_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_nope_attn_emb_fire_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=16 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"
# abacus
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_1_16_TBPTT_1024_batch_size_512_mask_before_equals_true_start_emb_abacus_attn_emb_nope_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=16 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=abacus

## FF
#nope
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_nope_attn_emb_nope_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.forward_only_model_with_skip=True
# fire
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_nope_attn_emb_fire_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire" arch.forward_only_model_with_skip=True
# abacus
python pretrain.py name=pos_or_one_vec_zeros_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_abacus_attn_emb_nope_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=1 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/or_one_vec_zeros/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=abacus arch.forward_only_model_with_skip=True

## FF w/ II
# nope
python pretrain.py name=pos_or_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_nope_attn_emb_nope_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/pos_arith_add_20_20_p_00/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.forward_only_model_with_skip=True
# fire
python pretrain.py name=pos_or_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_nope_attn_emb_fire_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/pos_arith_add_20_20_p_00/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"  arch.forward_only_model_with_skip=True
# abacus
python pretrain.py name=pos_or_bucket_20_20_reverse_all_pad_00_depthrec_16_1_TBPTT_1024_batch_size_256_mask_before_equals_true_start_emb_abacus_attn_emb_nope_with_skip_connections_run_1 wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=256 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=16 arch.maximal_recurrence=1 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/pos_arith_add_20_20_p_00/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True arch.embedding.pos_embedding=abacus arch.forward_only_model_with_skip=True

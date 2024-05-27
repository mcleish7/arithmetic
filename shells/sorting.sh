# REMINDER SET BASE DIR


## fire reverse
## fire reverse recall
## fire reverse recurrence

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_fire_8x1_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.attention.type='self-attention' \
	arch.attention.rotary_embedding='fire' impl.fullgraph=false impl.save_every_n_minutes=60 impl.save_intermediate_model_name='last'

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_fire_recall_8x1_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.attention.type='self-attention' \
	arch.attention.rotary_embedding='fire' impl.fullgraph=false impl.save_every_n_minutes=60 impl.save_intermediate_model_name='last' arch.forward_only_model_with_skip=True

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_fire_1x8_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=8 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.attention.type='self-attention' \
	arch.attention.rotary_embedding='fire' impl.fullgraph=false impl.save_every_n_minutes=60 impl.save_intermediate_model_name='last'

## abacus reverse
## abacus reverse recall
## abacus reverse recurrence

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_8x1_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus"

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_8x1_skip_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus" arch.forward_only_model_with_skip=True

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_1x8_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=8 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus"


## abacus fire reverse
## abacus fire reverse recall
## abacus fire reverse recurrence

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_with_fire_8x1_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus" \
	arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_with_fire_8x1_skip_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=8 arch.maximal_recurrence=1 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus" \
	arch.forward_only_model_with_skip=True arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"

torchrun --nproc_per_node=1 --standalone pretrain.py name=sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all_abacus_with_fire_1x8_1_24_run_1 \
	wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir \
	impl.microbatch_size=32 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=1 arch.maximal_recurrence=8 \
	arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True \
	data.sources.arithmetic.tokenized_dataset_path='arithmetic_data/sort_bucket_uniform_distribution_max_digits_n_10_max_length_m_10_20000000_p_00_reverse_all/hf_tokenized_dataset' \
	train.optim.lr=0.0001 arch.embedding.pos_embedding=None data.sources.arithmetic.tokenizer_type='sort' arch.mask_before_equals=True arch.embedding.pos_embedding="abacus" \
	arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"
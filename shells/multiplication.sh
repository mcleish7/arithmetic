## only Looped Transformer experiments for multiplication
torchrun --nproc_per_node=8 --standalone pretrain.py name=mul_bucket_15_15_reverse_all_pad_00_depthrec_4_4_TBPTT_1024_nope_mask_before_equals_batch_512_fire_abacus_8_gpu wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=4 arch.maximal_recurrence=4 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/x_bucket_method_n_15_m_15_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.00006 data.sources.arithmetic.tokenizer_type="pad" arch.attention.type="self-attention" arch.attention.rotary_embedding="fire" arch.mask_before_equals=True impl.fullgraph=false arch.loss_reduction=none arch.throttle=True arch.embedding.pos_embedding="abacus"

torchrun --nproc_per_node=8 --standalone pretrain.py name=mul_bucket_15_15_reverse_all_pad_00_depthrec_4_4_TBPTT_1024_nope_mask_before_equals_batch_512_fire_nope_8_gpu wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=4 arch.maximal_recurrence=4 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/x_bucket_method_n_15_m_15_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.00006 data.sources.arithmetic.tokenizer_type="pad" arch.attention.type="self-attention" arch.attention.rotary_embedding="fire" arch.mask_before_equals=True impl.fullgraph=false arch.loss_reduction=none arch.throttle=True arch.embedding.pos_embedding=None

torchrun --nproc_per_node=8 --standalone pretrain.py name=mul_bucket_15_15_reverse_all_pad_00_depthrec_4_4_TBPTT_1024_nope_mask_before_equals_batch_512_abacus_8_gpu wandb=none arch=crammed-depthrecurrent data=arithmetic base_dir=$cramming_base_dir impl.microbatch_size=512 budget=24 impl.compile_torch=False arch.objective_layout=TBPTT arch.layers_in_recurrent_block=4 arch.maximal_recurrence=4 arch.hidden_size=1024 arch.intermed_size=2048 impl.forbid_dataset_preprocessing=False impl.save_intermediate_checkpoints=True impl.save_final_model=True data.sources.arithmetic.tokenized_dataset_path="arithmetic_data/x_bucket_method_n_15_m_15_20000000_p_00_reverse_all/hf_tokenized_dataset" train.optim.lr=0.0001 data.sources.arithmetic.tokenizer_type="pad" arch.mask_before_equals=True impl.fullgraph=false arch.loss_reduction=none arch.throttle=True arch.embedding.pos_embedding="abacus"
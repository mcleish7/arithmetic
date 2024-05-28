# Transformers Can Do Arithmetic with the Right Embeddings! [Link to arXiv paper](https://arxiv.org/abs/2405.17399)

A joint project by: Sean McLeish, Arpit Bansal, Alex Stein,  Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild and Tom Goldstein



This repository contains code to replicate our research. It is a fork of the language model training framework [cramming](https://github.com/JonasGeiping/cramming) edited to for a next token prediction objective.

We provide a standalone implementation of Abacus Embeddings in [abacus.py](abacus.py).

## Citing Our Work
To cite our work, please use this bibtex.
```
@article{mcleish2024transformers,
    title={Transformers Can Do Arithmetic with the Right Embeddings}, 
    author={Sean McLeish and Arpit Bansal and Alex Stein and Neel Jain and John Kirchenbauer and Brian R. Bartoldson and Bhavya Kailkhura and Abhinav Bhatele and Jonas Geiping and Avi Schwarzschild and Tom Goldstein},
    journal={arXiv preprint arXiv:2405.17399},
    year={2024}
}
```

# Getting Started
We developed in Python 3.10.4, to install run:
```
git clone git@github.com:mcleish7/arithmetic.git
cd arithmetic
pip install .
```

On some machines you will need to run:
1. `pip install multiprocess -U`
2. `pip install dill -U`
3. `pip install apache-beam -U`

# Arithmetic
## Datasets
We release our datasets on [Google Drive](https://drive.google.com/drive/folders/1DqjCrUM1cNV7069Zl25_qBw2Px2xAw9j?usp=sharing) both in zipped format. We recommend you work with the zipped version until it is correctly placed in your file system.

Alternatively, you can make your own datasets using [create_data_split.py](create_data_split.py) using the commands from [shells/generate_and_tokenize_data.sh](shells/generate_and_tokenize_data.sh).

## File Structure
We recommend creating another directory `cramming-data` inside of arithmetic. This is where the models, logs and data will be stored.

You can either export you cramming base directory path to your `.bashrc` or you can replace `$cramming_base_dir` manually in the provided shells.
```
cd arithmetic
mkdir cramming-data
echo 'export cramming_base_dir=MY_BASE_DIR' >> ~/.bashrc
source ~/.bashrc
```
For example, this may look like: `echo 'export cramming_base_dir=~/arithmetic/cramming-data' >> ~/.bashrc`

For example our file system looks like:
```
cramming-generative
└── cramming-data
    ├── addition-train-one
    │    ├── pretrain/<DATE>/<TIME>
    │    │    ├── .hydra
    │    │    │   ├── config.yaml
    │    │    │   ├── hydra.yaml
    │    │    │   └── overrides.yaml
    │    │    └── addition-train-one_pretrain.log
    │    ├── checkpoints/FINAL_<LOSS_VAL>
    │    │    ├── model_config.json
    │    │    ├── model.safetensors
    │    │    └── state_dict.pth
    │    └── downstream
    └── data
        └── arithmetic_data
            ├── +_grid_eval_dataset_reverse_all_tokenized
            └── ... other datasets ...
```

## Training
Example commands are in the [shells](shells) directory, organised by task.

### Explanation of Some Commands
1. Give samples instead of tokens equal importance in loss: `arch.loss_reduction=none`
2. Divide the gradients in the recurrent block by the number of recurrences: `arch.throttle=True`
3. Mask before the equals sign: `arch.mask_before_equals=True`
4. Skip connections inside of the recurrent block: `arch.forward_only_model_with_skip=True`
5. Multi-GPU: `python` -> `torchrun --nproc_per_node=<NUM GPUS> --standalone ` and add `impl.fullgraph=false`

### Positional Embeddings:
#### Absolute
1. Learned: `arch.embedding.pos_embedding=learned`
2. Abacus: `arch.embedding.pos_embedding=abacus`
* If you want the maximum k in abacus to be larger: `arch.embedding.max_abacus_len=100`, be default this value is 100. Abacus is also implemented in a standalone manner in [abacus.py](abacus.py).

#### Relative
1. NoPE: `arch.embedding.pos_embedding=None`
2. FIRE: `arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire"`
3. FIRE randomised: e.g:`arch.embedding.pos_embedding=None arch.attention.type="self-attention" arch.attention.rotary_embedding="fire" arch.attention.max_length=128` by default `arch.attention.max_length=0` so setting this longer than the max sequence length gives some randomness in the embedding.
4. RoPE: `arch.attention.type="self-attention" arch.attention.rotary_embedding=true`

### Checkpointing
We have implemented *single* GPU training checkpointing, to do this use:
`impl.save_every_n_minutes=60 impl.save_intermediate_model_name='last'`
This saves a checkpoint every 60 minutes under the name 'last'

Caution: This feature is not fully tested for multi-GPU cases. We also cannot currently train models which have used their full budget for longer.

### WandB
You can log runs to your weights&biases account. To do so, simply modify `wandb.entity` and `wandb.project` on the command line or at [cramming/config/wandb/default.yaml](cramming/config/wandb/default.yaml).

## Testing
We show examples in [shells/evaluation.sh](shells/evaluation.sh). 

We provide a very basic automation in [gen_eval_script.py](gen_eval_script.py), this prints the basic commands you may need to further edit these.

### Addition
For addition we have a very large possible evaluation set, we do a grid search over a 100x100 grid which we split into 20 pieces with the aim of balancing the number of forward calls across all 20 pieces.
We then have a further eval for operand lengths 100->160.

### Multiplication
We only evaluate up to 25x25, which we do in a single job.

### Sorting
Sorting uses a separate evaluation file [sort_eval.py](sort_eval.py), this is because the evaluation calls cannot be parallelised, making evaluation much longer.
The evaluation cannot be parallelised because the place of the equals sign is not fixed for a batch.
We currently evaluate across 30 jobs for a 30x30 grid but this can be reduced to a smaller number of jobs using these flags: `max_size_given, start_ind_1_given, start_ind_2_given`

### Bitwise OR
We use the same framework as for addition but the process is quicker as some of the batches do not contain 100 samples as there are not 100 possibilities for some batches. Unlike addition we do not sample with replacement for this task.

# Analysis
1. We provide [pretty_plotter.py](pretty_plotter.py) to combine the small evaluation grids together into one plot.
Use this by putting the model name into the string at the top of the `main` function.
2. For the large 100x100 grids we provide [pretty_plotter_big.py](pretty_plotter_big.py).
These are designed to be as flexible as possible but may need to be edited to fit your file set up.
3. For sorting, we provide [pretty_plotter_sort.py](pretty_plotter_sort.py), this allows us to read the individual `.txt` files created during testing and merge them all together into a nice plot.

# Contact
Please, feel free to contact us with any questions, or open an issue on Github.
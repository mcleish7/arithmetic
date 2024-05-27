## Training Data -- these commands approximately correspond to the zipped data we provide

# bitwise or
python create_pos_or_variants.py --n 20 --m 20 --dir_name <NAME> --max 100
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.01

# addition
python create_data_split.py --bucket --op + --n 20 --m 20 --limit 20000000 --p 0.0 --dir_name <NAME> --reverse_all
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.01

# addition with index hints
python create_data_split.py --bucket --op + --n 20 --m 20 --limit 20000000 --p 0.0 --dir_name <NAME> --reverse_all --index_hints
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type index

# multiplication
python create_data_split.py --bucket --op x --n 15 --m 15 --limit 20000000 --dir_name <NAME>  --reverse_all --p 0.0
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.01

# sorting
python create_data_split.py --uniform_distribution_sort_data --continue_to_tokenize --tokenize --tokenizer_type sort --test_split_ratio 0.01 --n 10 --m 10 --limit 20000000 --dir <NAME> --sort_generation_method bucket_uniform_distribution --reverse_all

## Evaluation Data -- run line and tokenize once for each operand length
# bitwise or
python create_pos_or_variants.py --n <i> --m <j> --dir_name <NAME> --exact --eval --max 100
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.0

# addition
python create_data_split.py --op + --n <i> --m <j> --num_samples 100 --dir_name <NAME> --exact
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.0

# multiplication
python create_data_split.py --op x --n <i> --m <j> --num_samples 100 --dir_name <NAME> --exact
python create_data_split.py --tokenize --dir_name <NAME> --tokenizer_type pad --test_split_ratio 0.0

# sorting
python create_data_split.py --uniform_distribution_sort_data --continue_to_tokenize --tokenize --tokenizer_type sort --test_split_ratio 0.01 --n <i> --m <j> --limit 100 --dir <NAME> --sort_generation_method bucket_uniform_distribution --reverse_all --exact
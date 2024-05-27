"""Prepare and preprocess datasets."""

import torch
import datasets
import hydra
import pandas as pd
import os
import contextlib
import logging
import tempfile
from itertools import chain
from collections import defaultdict

import json
from omegaconf import OmegaConf

from .tokenizer_preparation import construct_tokenizer, load_tokenizer
from .curriculum_sorting import _sort_tokenized_dataset_by_unigram, _sort_tokenized_dataset_by_token, _sort_tokenized_dataset_by_word_length
from .deduplicate import deduplicate_huggingface_dataset
from .utils import checksum_config, stage_dataset, detailed_OSError
from .tokenizer_preparation import get_tokenizer


import random
import transformers

from datasets.distributed import split_dataset_by_node
import random

from torch.utils.data import DataLoader
from typing import Dict


log = logging.getLogger(__name__)
datasets.enable_progress_bar()
datasets.disable_caching()  # We'll save only the final preprocessed dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_num_workers(cfg_impl):
    if cfg_impl is None:
        return 0
    elif cfg_impl.threads > 0:
        return min(torch.get_num_threads() // max(1, torch.cuda.device_count()), cfg_impl.threads)
    else:
        return 0


def load_pretraining_corpus(cfg_data, cfg_impl, data_dir: str = None):
    """Load (and optionally stage) a pre-processed corpus. Create one if it doesn't exist."""
    datasets.disable_caching()
    checksum = checksum_config(cfg_data)

    data_path = data_dir
    if data_path is None:
        data_path = cfg_impl.path
    data_src = list(cfg_data.sources.values())[0]
    provider = data_src["provider"]
    tokenizer_type = data_src["tokenizer_type"]
    if provider == "fake":
        # Shortcut for fake data
        return _load_fake_dataset(cfg_data, data_src, path=cfg_impl.path)
    elif provider == "hub":
        # pulling from huggingface
        return _load_from_hub(cfg_data, data_path)
    elif provider == "arithmetic":
        # our math data
        tokenized_dataset_path = data_src["tokenized_dataset_path"]
        tokenized_dataset_path = os.path.join(data_path, tokenized_dataset_path)
        print(f"Loading tokenized dataset from {tokenized_dataset_path}")
        tokenized_data = load_tokenized_data(tokenized_dataset_path)
        print(f"Loaded tokenized dataset from {tokenized_dataset_path}")
        tokenizer = get_tokenizer(tokenizer_type)
        print(f"Loaded tokenizer {tokenizer_type}")
        tokenizer.model_max_length = cfg_data["seq_length"]  # not perfect but better than nothing
        return tokenized_data, tokenizer
    else:
        # not found so creating
        try:
            if cfg_impl.local_staging_dir is not None:
                with main_process_first():
                    data_path = stage_dataset(data_path, cfg_impl.local_staging_dir)
            # Load already processed dataset
            tokenized_dataset = datasets.load_from_disk(data_path)
            tokenizer = load_tokenizer(
                os.path.join(data_path, "tokenizer"),
                seq_length=cfg_data.seq_length,
                vocab_size=cfg_data.vocab_size,
                cache_dir=cfg_impl.path,
            )
        except FileNotFoundError:
            if cfg_impl.forbid_dataset_preprocessing:
                raise ValueError(
                    f"Cannot find processed at path {data_path}. Dataset preprocessing disabled. "
                    "Dataset preprocessing can be enabled with 'impl.forbid_dataset_preprocessing=False'."
                )
            # Run preprocessing to create dataset
            with main_process_first():
                num_threads = min(torch.get_num_threads(), cfg_impl.threads)  # Mitigate worker overloading
                preprocessed_dataset, new_tokenizer = preprocess_dataset(
                    cfg_data,
                    download_path=cfg_impl.path,
                    num_threads=num_threads,
                    max_raw_chunk_size=cfg_impl.max_raw_chunk_size,
                )

                def save_corpus(path):
                    preprocessed_dataset.save_to_disk(path)
                    new_tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
                    with open(os.path.join(path, "model_config.json"), "w") as file:
                        json.dump(OmegaConf.to_container(cfg_data, resolve=True), file)

                if not cfg_impl.temporary_corpus:
                    # Save to base directory:
                    save_corpus(os.path.join(cfg_impl.path, processed_dataset_dir))
                    if cfg_impl.local_staging_dir is not None:
                        # Optionally also copy into local staging directory
                        data_path = stage_dataset(data_path, cfg_impl.local_staging_dir)
                else:
                    # Directly use staging directory
                    save_corpus(os.path.join(cfg_impl.local_staging_dir, processed_dataset_dir))

            # Reload dataset
            tokenized_dataset = datasets.load_from_disk(data_path)
            tokenizer = load_tokenizer(
                os.path.join(data_path, "tokenizer"),
                seq_length=cfg_data.seq_length,
                vocab_size=cfg_data.vocab_size,
                cache_dir=cfg_impl.path,
            )

    # Cast to tensors after loading from arrow:
    tokenized_dataset.set_format("torch")

    # 4) Log overviews so we always know what's going on with weird tokenization tricks
    dataset_size = len(tokenized_dataset["train"])
    random_sentence_idx = torch.randint(0, dataset_size, (1,)).item()
    input_data = tokenized_dataset["train"][random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim

    log.info(f"Random sentence with seq_length {tokenizer.model_max_length} from dataset of size {dataset_size:,}: ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])
    log.info("above is tokenized into below with _ joined to every token")
    log.info("_".join(tokenizer.decode(t) for t in input_data))
    return tokenized_dataset, tokenizer

def load_tokenized_data(tokenized_dataset_path):
    tokenized_dataset = datasets.load_from_disk(tokenized_dataset_path)
    return tokenized_dataset

def convert_to_hf_dataset(tokenized_data):
    # Convert the PyTorch tensor to a list of lists (if it's not already)
    data_list = tokenized_data.tolist()

    # Create a DataFrame from the list
    df = pd.DataFrame({'tokens': data_list})

    # Convert the DataFrame to a Hugging Face dataset
    hf_dataset = datasets.Dataset.from_pandas(df)
    return hf_dataset

def preprocess_dataset(cfg_data, download_path, num_threads=1, max_raw_chunk_size=1e14):
    """A lot of loading and preprocessing."""
    # 1) Collect raw source datasets
    raw_datasets = []
    for name, details in cfg_data.sources.items():
        log.info(f"Now preparing source {name}...")
        if details.provider == "huggingface":
            if name == "EleutherAI/proof-pile-2":
                raw_dataset = datasets.load_dataset(
                    name,
                    name=details.partition,
                    split=details.split,
                    cache_dir=download_path,
                    streaming=details.streaming,
                )
            else:              
                raw_dataset = datasets.load_dataset(
                    name,
                    data_dir=details.partition,
                    split=details.split,
                    cache_dir=download_path,
                    streaming=details.streaming,
                )
        elif details.provider == "local":
            raw_dataset = datasets.load_dataset(details.file_type, data_files=details.files, streaming=details.streaming)[details.split]
        else:
            raise ValueError(f"Invalid data provider {details.provider} given.")

        # remove columns that break later processing steps
        if details.remove_columns is not None:
            raw_dataset = raw_dataset.remove_columns(details.remove_columns)
        # Filter?
        if getattr(details, "filter", None) is not None:

            def filter_fn(entry):
                """Assume a metadata key 'meta' is present"""
                for key, values in details.filter.items():
                    if entry["meta"][key] in values:
                        return True
                return False

            raw_dataset = raw_dataset.filter(filter_fn)
        # move streams to fixed datasets to make everything sane (and to allow concatenation with unstreamed data)
        if details.streaming:
            raw_dataset = raw_dataset.take(int(cfg_data.max_entries_in_raw_dataset))
            raw_dataset = _move_stream_to_fixed_map(raw_dataset, cfg_data.max_entries_in_raw_dataset, max_raw_chunk_size)
        else:
            if cfg_data.max_entries_in_raw_dataset < len(raw_dataset):
                raw_dataset = raw_dataset.select(range(int(cfg_data.max_entries_in_raw_dataset)))
        # concatenate dataset that were cut into pieces that are too small
        if details.concatenate_successive_entries > 0:
            raw_dataset = _concatenate_entries(raw_dataset, details.concatenate_successive_entries, num_threads=num_threads)
        raw_datasets += [raw_dataset]

    # 2) Preprocess and tokenize
    raw_data = datasets.concatenate_datasets(raw_datasets)
    raw_data = raw_data.shuffle(seed=89)  # Shuffle once here so that multiproc has shards of similar size!
    # This shuffle is crucial for fast multiprocessing tokenization
    # because datasets.map uses a contiguous sharding under the hood.

    # However, we also shuffle so we can now select a smaller range:
    if cfg_data.max_entries_in_raw_dataset < len(raw_data):
        raw_data = raw_data.select(range(int(cfg_data.max_entries_in_raw_dataset)))

    raw_data = raw_dataset_preprocessing(raw_data, num_threads, cfg_data)  # This is by default a no-op, but can be dedup, filtering...
    tokenizer = construct_tokenizer(raw_data, cfg_data, path=download_path)
    tokenized_dataset = _huggingface_preprocessing(raw_data, tokenizer, cfg_data, num_threads=num_threads)  # Tokenize, group, sort...

    return tokenized_dataset, tokenizer


def _move_stream_to_fixed_map(raw_data_streamed, max_entries_in_raw_dataset, max_raw_chunk_size=1e14):
    """Save streaming dataset to a fixed mapping-style database."""
    # I'm tired of IterableDatasets and will take the performance hit to write them out instead:
    try:
        if max_raw_chunk_size > max_entries_in_raw_dataset:
            with tempfile.TemporaryDirectory() as tmpdirname:
                datasets.Dataset.from_dict(dict(text=[v["text"] for v in raw_data_streamed])).save_to_disk(tmpdirname + "raw_data")
                raw_data_mapped = datasets.load_from_disk(tmpdirname + "raw_data")
            # This used to be only a move into RAM but this breaks memory later using C4:
            # raw_data = datasets.Dataset.from_dict(dict(text=[v["text"] for v in raw_data]))
            return raw_data_mapped
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                mapped_sets = []
                data_in_RAM = defaultdict(list)
                for idx, value_stream in enumerate(raw_data_streamed):
                    data_in_RAM["text"].append(value_stream["text"])
                    if ((idx + 1) % max_raw_chunk_size == 0) or ((idx - 1) == max_entries_in_raw_dataset):
                        datasets.Dataset.from_dict(data_in_RAM).save_to_disk(tmpdirname + "raw_data" + str(idx))
                        mapped_dataset = datasets.load_from_disk(tmpdirname + "raw_data" + str(idx))
                        log.info(
                            f"Saved temporary copy at idx {idx} of {max_entries_in_raw_dataset} at {tmpdirname + 'raw_data' + str(idx)}."
                        )
                        data_in_RAM["text"] = []
                        mapped_sets.append(mapped_dataset)
            return datasets.concatenate_datasets(mapped_sets)
    except OSError as e:
        detailed_OSError(e)


def _huggingface_preprocessing(raw_dataset, tokenizer, cfg_data, num_threads=4):
    """Dataset preprocessing and tokenization.

    This is basically the default HF routine from
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
    """
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer.model_max_length
    map_setup = dict(
        batched=True,
        batch_size=512,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=False,
    )
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # The Collator is modified not to read special_masks anyway:

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            return_special_tokens_mask=False,
            return_attention_mask=False,  # handle this manually elsewhere if necessary
            return_token_type_ids=False,
        )

    tokenizer.model_max_length = 1e30
    tokenized_dataset = raw_dataset.map(
        tokenize_function, remove_columns=column_names, desc="Running tokenizer on every text in dataset", **map_setup
    )
    tokenizer.model_max_length = max_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] for k, t in concatenated_examples.items()}
        return result

    tokenized_dataset = tokenized_dataset.map(group_texts, desc=f"Grouping texts in chunks of {max_seq_length}", **map_setup)

    # Reduce size to maximal limit:
    if cfg_data.max_seq_in_tokenized_dataset < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(int(cfg_data.max_seq_in_tokenized_dataset)), keep_in_memory=True)

    # Split into train-val
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=cfg_data.validation_seqs, shuffle=False)

    # Shuffle?
    if cfg_data.ordering == "randomized":
        tokenized_dataset["train"] = tokenized_dataset["train"].shuffle(seed=233)
    elif cfg_data.ordering == "unigram-curriculum":
        tokenized_dataset["train"] = _sort_tokenized_dataset_by_unigram(tokenized_dataset["train"], tokenizer, num_threads)
    elif cfg_data.ordering == "word-length-curriculum":
        tokenized_dataset["train"] = _sort_tokenized_dataset_by_word_length(tokenized_dataset["train"], tokenizer, num_threads)
    elif cfg_data.ordering == "sentence-length-curriculum":
        tokenized_dataset["train"] = _sort_tokenized_dataset_by_token(
            tokenized_dataset["train"],
            tokenizer,
            tokenizer.vocab[" ."],
            num_threads,
        )
    elif cfg_data.ordering == "fragment-curriculum":
        tokenized_dataset["train"] = _sort_tokenized_dataset_by_token(
            tokenized_dataset["train"],
            tokenizer,
            tokenizer.vocab["<eot>"],
            num_threads,
        )
    else:
        raise ValueError(f"Invalid dataset ordering {cfg_data.ordering} provided.")

    # Finally flatten
    # This is necessary for the save_to_disk call that comes next. If skipped here, the call will be invoked from save_to_disk
    # This way, atleast it shares the same batch parameters and prints a progress bar.
    tokenized_dataset = tokenized_dataset.map(desc="Flattening the indices", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return tokenized_dataset


def _load_fake_dataset(cfg_data, details, path=None):
    tokenizer = load_tokenizer(cfg_data.tokenizer, cfg_data.seq_length, cfg_data.vocab_size, cache_dir=path)
    tokenizer.model_max_length = cfg_data.seq_length
    generator = torch.Generator()
    generator.manual_seed(details.randgen_seed)
    dataset = torch.randint(0, cfg_data.vocab_size, (details.size, cfg_data.seq_length), generator=generator)
    return dataset, tokenizer


def _concatenate_entries(dataset, num_entries_in_group, num_threads):
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def group_texts(examples):
        result = dict()
        for key, entries in examples.items():
            reduced_list = []
            state, num_collected = None, 0
            for entry in entries:
                num_collected += 1
                if num_collected == 1:
                    state = entry
                else:
                    state += entry
                if num_collected == num_entries_in_group:
                    reduced_list.append(state)
                    state, num_collected = None, 0

            result[key] = reduced_list

        return result

    map_setup = dict(
        batched=True,
        batch_size=512,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=True,
    )
    dataset = dataset.map(group_texts, desc="Concatenating examples", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return dataset


def raw_dataset_preprocessing(raw_dataset, num_threads, cfg_data):
    """Some dataset "improvements". These are optional filtering or normalization rules that are only applied to the pretraining corpus.
    This separates them from generic normalizations that are baked into the tokenizer."""
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]
    known_tokens = []
    map_setup = dict(
        batched=True,
        batch_size=512,
        num_proc=None,  # a bit messy but c4 in RAM can be overbearing otherwise
    )
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg_data.remove_trash:
        # experimental first test based on Unigram tokenization:
        from transformers import AutoTokenizer

        if cfg_data.remove_trash == "self":
            os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
            tokenizer = construct_tokenizer(raw_dataset, cfg_data, path=None)
            if num_threads > 0:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        tokenizer.model_max_length = 1e30

        def filtering_rule(examples):
            tokenized = tokenizer(examples[text_column_name])["input_ids"]
            return [len(t) < cfg_data.trash_cutoff * len(e) for t, e in zip(tokenized, examples[text_column_name])]

        log.info(f"Size of dataset before trash removal: {len(raw_dataset)}.")
        raw_dataset = raw_dataset.filter(
            filtering_rule,
            desc="Filter sentences that cannot be tokenized well.",
            **map_setup,
        )
        log.info(f"Size of filtered dataset: {len(raw_dataset)}.")

    if cfg_data.deduplicate_entries:
        log.info(f"Size of dataset before deduplication: {len(raw_dataset)}.")
        raw_dataset = deduplicate_huggingface_dataset(
            raw_dataset, threshold=cfg_data.deduplication_threshold, original_cwd=hydra.utils.get_original_cwd()
        )
        log.info(f"Size of deduplicated dataset: {len(raw_dataset)}.")

    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return raw_dataset


@contextlib.contextmanager
def main_process_first():
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    This is a stripped-down version of the the huggingface context manager from commit 2eb7bb15e771f13192968cd4657c78f76b0799fe
    """
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield


def _load_from_hub(cfg_data, data_path):
    from huggingface_hub import hf_hub_download

    tokenized_dataset = datasets.load_dataset(cfg_data.hf_location, "train", streaming=cfg_data.streaming, cache_dir=data_path)["train"]
    tokenized_dataset = tokenized_dataset.with_format("torch")

    tokenizer_req_files = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    os.makedirs(os.path.join(data_path, "tokenizer"), exist_ok=True)
    for file in tokenizer_req_files:
        hf_hub_download(
            cfg_data.hf_location,
            file,
            subfolder="tokenizer",
            repo_type="dataset",
            local_dir=os.path.join(data_path),
        )
    tokenizer = load_tokenizer(os.path.join(data_path, "tokenizer"), seq_length=cfg_data.seq_length, cache_dir=data_path)
    return tokenized_dataset, tokenizer


def prepare_dataloaders(datasets, tokenizer, cfg_train, cfg_impl) -> Dict[str, DataLoader]:
    dataloaders = dict()
    train_loader = prepare_pretraining_dataloader(datasets["train"], tokenizer, cfg_train, cfg_impl)
    dataloaders["train"] = train_loader
    dataloaders["test"] = prepare_validation_dataloader(datasets["test"], tokenizer, cfg_impl)
    return dataloaders


def prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl) -> torch.utils.data.DataLoader:

    num_workers = get_num_workers(cfg_impl)
    collate_fn = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)

    if dataset is None:
        # generate data at runtime
        return RuntimeInfiniteDataLoader(tokenizer, device)
    elif isinstance(dataset, torch.utils.data.IterableDataset):
        # streaming mode for ready-made datasets, speed not tested
        if torch.distributed.is_initialized():
            dataset = split_dataset_by_node(dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

        if cfg_impl.shuffle_in_dataloader:
            dataset = dataset.shuffle(seed=42, buffer_size=256)
        if cfg_train.reverse_dataset_order:
            raise ValueError("Reverse stream not implemented.")
        sampler = None
    else:
        # Normally, we'd just use nice map-style datasets:
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=cfg_impl.shuffle_in_dataloader,
                drop_last=True,
            )
        else:
            if cfg_impl.shuffle_in_dataloader:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

    if cfg_train.reverse_dataset_order:
        dataset = dataset.select(reversed(range(len(dataset))))
    repeated_dataloader = InfiniteDataLoader(
        dataset,
        sampler=sampler,
        batch_size=min(cfg_impl.microbatch_size, len(dataset)),
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        drop_last=True,
        prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
        persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    return repeated_dataloader


def prepare_validation_dataloader(dataset, tokenizer, cfg_impl):

    num_workers = get_num_workers(cfg_impl)
    collate_fn = FastDataCollatorForLanguageModeling(tokenizer=tokenizer, pad_to_multiple_of=8, mlm=False)
    if dataset is None:
        # generate data at runtime
        return RuntimeInfiniteDataLoader(tokenizer, device)
    elif isinstance(dataset, torch.utils.data.IterableDataset):
        sampler = None
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=min(cfg_impl.microbatch_size, len(dataset)),
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        drop_last=True,  # better make it fit elsewhere
        prefetch_factor=cfg_impl.prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    return dataloader


"""This is a minor modification of huggingface's toking masking:"""
"""original source:
https://github.com/huggingface/transformers/blob/130b987880a9b1ade5c76dc1413c12c8924fda50/src/transformers/data/data_collator.py#L748
at commit f00f22a3e290fd377b979124dcf9800b3d73eb11"""


class FastDataCollatorForLanguageModeling(transformers.DataCollatorForLanguageModeling):
    def __init__(self, *args, create_labels_entry=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm = False
        self.create_labels_entry = create_labels_entry

    def torch_call(self, examples):
        """Simplified call assuming all dicts in the list of examples have the same layout and contain tensors.
        Assume further that all these tensors contain vectors of Long Tensors  [AND THEY HAVE TO BE LONG]"""
        if isinstance(examples[0], torch.Tensor):
            examples = [{"input_ids": ex} for ex in examples]
        # So this is the handmade version
        batch = dict()
        for key in examples[0].keys():
            elem = torch.as_tensor(examples[0][key])
            out = None
            if torch.utils.data.get_worker_info() is not None:
                storage = elem._typed_storage()._new_shared(len(examples) * elem.shape[0], device=elem.device)
                out = elem.new(storage).resize_(len(examples), elem.shape[0])

            batch[key] = torch.stack([torch.as_tensor(example[key]) for example in examples], 0, out=out).contiguous()

        if self.create_labels_entry:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """Lazy copy-paste from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()
        self.epoch_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            self.epoch_counter += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch_counter)
            batch = next(self.dataset_iterator)
        return batch

    def set_epoch(self, epoch: int):
        self.epoch_counter = epoch

class RuntimeInfiniteDataLoader(torch.utils.data.DataLoader):
    """Lazy copy-paste from https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958."""

    def __init__(self, tokenizer, device, *args, **kwargs):
        self.epoch_counter = 0
        ## All need to be moved to cfg
        self.max_n = 20
        self.max_m = 20
        self.batch_size = 16
        self.reverse_answer = False
        self.reverse_all = False
        self.operation = '+'

        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.vocab[self.tokenizer.eos_token]
        self.device = device
        self.current_batch = []

    def get_arithmetic(self, n, m):
        batch = []
        for _ in range(self.batch_size):
            num1 = random.randint((10**(n-1)), (10**n - 1))
            num2 = random.randint(10**(m-1), 10**m - 1)

            num1_str = str(num1)
            num2_str = str(num2)

            result = num1 + num2

            result = str(result)

            if self.reverse_answer:
                result = result[::-1]
            if self.reverse_all:
                result = result[::-1]
                num1_str = num1_str[::-1]
                num2_str = num2_str[::-1]

            batch.append(f"{num1_str}{self.operation}{num2_str}={result}")

        return batch

    def tokenize_batch(self, batch):
        # todo this can be sped up using the HF dataset.map
        tokenized_list = [self.tokenizer(entry)["input_ids"] + [self.eos_token_id] for entry in batch]

        max_length = max(len(entry) for entry in tokenized_list)
        pad_token_id = self.tokenizer.pad_token_id
        tokenized_list = [entry + [pad_token_id] * (max_length - len(entry)) for entry in tokenized_list]

        tokenized_tensor = torch.tensor(tokenized_list, device=self.device)
        return tokenized_tensor

    def __iter__(self):
        return self

    def __next__(self):
        n = random.randint(1, self.max_n)
        m = random.randint(1, self.max_m)
        batch = self.get_arithmetic(n, m)
        tokenized_batch = self.tokenize_batch(batch)
        return {'input_ids': tokenized_batch, 'max_recur': max(n, m)+5}

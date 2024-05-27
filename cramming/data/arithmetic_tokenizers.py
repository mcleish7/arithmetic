"""
Character level tokenizers for arithemtic projects
Multiple tokenizers for different tasks
"""

from transformers import PreTrainedTokenizer
import re
import torch
import random

class CustomCharLevelTokenizerForAddingPadding(PreTrainedTokenizer):
    """Simple char level math tokenizer"""
    def __init__(self, **kwargs):
        # Define the characters to tokenize
        characters = '0123456789+-x= '

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

        super().__init__(**kwargs)

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # Tokenize the text character by character
        # text = re.sub('\s+',' ',text)
        temp = [char if char in self.vocab else self.unk_token for char in text]
        temp = [item.replace(' ', '[PAD]') for item in temp]
        return temp

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        # Convert an ID to its corresponding token
        return self.ids_to_tokens.get(index, self.unk_token)

    def __call__(self, text, **kwargs):
        # Tokenize text and convert to input IDs
        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(token) for token in tokens]
        return {"input_ids": input_ids}

    def decode(self, token_ids, **kwargs):
        # Convert token IDs to tokens and join into a string
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return ''.join(tokens).replace(self.pad_token, '').replace(self.bos_token, '').replace(self.eos_token, '')


class CustomCharLevelTokenizerForAddingPaddingWithIndexHints(PreTrainedTokenizer):
    """Tokenizer for index hints"""
    def __init__(self, **kwargs):
        # Define the characters to tokenize
        characters = '0123456789+-x= '
        self.char_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwyz!@£#$%^&*()~?.,<>{}[]:;/|βΓΔδεζηθκΛλμΞξΠπΣςτΦφχΨψΩω"
        characters = characters + self.char_set

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

        super().__init__(**kwargs)

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}
        
    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # Tokenize the text character by character
        # text = re.sub('\s+',' ',text)
        temp = [char if char in self.vocab else self.unk_token for char in text]
        temp = [item.replace(' ', '[PAD]') for item in temp]
        return temp

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        # Convert an ID to its corresponding token
        return self.ids_to_tokens.get(index, self.unk_token)

    def __call__(self, text, **kwargs):
        # Tokenize text and convert to input IDs
        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(token) for token in tokens]
        return {"input_ids": input_ids}

    def decode(self, token_ids, **kwargs):
        # Convert token IDs to tokens and join into a string
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return ''.join(tokens).replace(self.pad_token, '').replace(self.bos_token, '').replace(self.eos_token, '')


class CustomCharLevelTokenizerSort(PreTrainedTokenizer):
    """Tokenizer for sorting"""
    def __init__(self, **kwargs):
        # Define the characters to tokenize
        characters = '0123456789D,:= '
        set_of_chars = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', '!', '@', '£', '#', '$', '%', '^',
                        '&', '*', '(', ')', '~', '?', '.', '<', '>', '{', '}', '[', ']', ';', '/', '|', 'β', 'Γ', 'Δ',
                        'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'Λ', 'λ', 'μ', 'Ξ', 'ξ', 'Π', 'π', 'Σ', 'ς', 'τ', 'Φ', 'φ', 'χ',
                        'Ψ', 'ψ', 'Ω', 'ω']
        self.char_set = ''.join(set_of_chars)
        characters = characters + self.char_set

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

        super().__init__(**kwargs)

        # Define and set special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'

        # Combine characters and special tokens to form the custom vocabulary
        self.vocab = {char: i + 4 for i, char in enumerate(characters)}  # Starting from 4 to account for special tokens
        self.vocab.update({self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3})

        # Create the reverse mapping from IDs to tokens
        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text):
        # Tokenize the text character by character
        temp = [char if char in self.vocab else self.unk_token for char in text]
        temp = [item.replace(' ', '[PAD]') for item in temp]
        return temp

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        # Convert an ID to its corresponding token
        return self.ids_to_tokens.get(index, self.unk_token)

    def __call__(self, text, **kwargs):
        # Tokenize text and convert to input IDs
        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(token) for token in tokens]
        return {"input_ids": input_ids}

    def decode(self, token_ids, **kwargs):
        # Convert token IDs to tokens and join into a string
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        return ''.join(tokens).replace(self.pad_token, '').replace(self.bos_token, '').replace(self.eos_token, '')

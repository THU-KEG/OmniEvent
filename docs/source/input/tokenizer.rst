Tokenizer
=========

.. code-block:: python

    import collections
    import logging
    import numpy as np
    import os
    import pdb

    from transformers import PreTrainedTokenizer
    from typing import Dict, Iterable, List, Optional, Tuple, Union

    logger = logging.getLogger(__name__)

``load_vocab``
----------------

Loads a vocabulary file, allocates a unique id for each word within the vocabulary and saves the correspondence
between words and ids into a dictionary. Generates and returns word embeddings if it is required.

**Args:**

- ``vocab_file``: The path of the vocabulary file.
- ``return_embeddings``: Whether or not to return the word embeddings.

**Returns:**

- ``word_embeddings``: An numpy array represents each word's embedding within the vocabulary, with the size of (number of words) * (embedding dimension). Returns word embeddings if ``return_embeddings`` is set as ``True``.
-  ``vocab``: A dictionary indicates the unique id of each word within the vocabulary.

.. code-block:: python

    def load_vocab(vocab_file: str,
                return_embeddings: bool = False) -> Union[Dict[str, int], np.ndarray]:
        """Loads a vocabulary file into a dictionary.

        Loads a vocabulary file, allocates a unique id for each word within the vocabulary and saves the correspondence
        between words and ids into a dictionary. Generates and returns word embeddings if it is required.

        Args:
            vocab_file (`str`):
                The path of the vocabulary file.
            return_embeddings (`bool`, `optional`, defaults to `False`):
                Whether or not to return the word embeddings.

        Returns:
            word_embeddings (`np.ndarray`):
                An numpy array represents each word's embedding within the vocabulary, with the size of (number of words) *
                (embedding dimension). Returns word embeddings if `return_embeddings` is set as True.
            vocab (`Dict[str, int]`):
                A dictionary indicates the unique id of each word within the vocabulary.
        """
        vocab = collections.OrderedDict()
        vocab["[PAD]"] = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            lines = reader.readlines()
        num_embeddings = len(lines) + 1
        embedding_dim = len(lines[0].split()) - 1
        for index, line in enumerate(lines):
            token = " ".join(line.split()[:-embedding_dim])
            if token in vocab:
                token = f"{token}_{index+1}"
            vocab[token] = index + 1
        if return_embeddings:
            word_embeddings = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
            for index, line in enumerate(lines):
                embedding = [float(value) for value in line.strip().split()[-embedding_dim:]]
                word_embeddings[index+1] = embedding
            return word_embeddings
        return vocab

``whitespace_tokenize()``
-------------------------

Cleans the whitespace at the beginning and end of the text and splits the text into a list based on whitespaces.

**Args:**
- ``tex``: A string representing the input text to be processed.

**Returns:**

- ``tokens``: A list of strings in which each element represents a word within the input text.

.. code-block:: python

    def whitespace_tokenize(text: str) -> List[str]:
        """Runs basic whitespace cleaning and splitting on a piece of text.

        Cleans the whitespace at the beginning and end of the text and splits the text into a list based on whitespaces.

        Args:
            text (`str`):
                A string representing the input text to be processed.

        Returns:
            tokens (`List[str]`):
                A list of strings in which each element represents a word within the input text.
        """
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

``WordLevelTokenizer``
----------------------

This tokenizer inherits from ``PreTrainedTokenizer`` which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

**Attributes:**

- ``vocab``: A dictionary indicating the correspondence between words and ids within the vocabulary.
- ``ids_to_tokens``: A dictionary indicating the correspondence between ids and words within the vocabulary.
- ``whitespace_tokenizer``: A ``WhitespaceTokenizer`` instance for word piece tokenization.

.. code-block:: python

    VOCAB_FILES_NAMES = {"vocab_file": "vec.txt"}

    PRETRAINED_VOCAB_FILES_MAP = {}

    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}

    PRETRAINED_INIT_CONFIGURATION = {}

.. code-block:: python

    class WordLevelTokenizer(PreTrainedTokenizer):
        """Construct a BERT tokenizer. Based on WordPiece.

        This tokenizer inherits from `PreTrainedTokenizer` which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods.

        Attributes:
            vocab (`Dict[str, int]`):
                A dictionary indicating the correspondence between words and ids within the vocabulary.
            ids_to_tokens (`Dict[int, str]`):
                A dictionary indicating the correspondence between ids and words within the vocabulary.
            whitespace_tokenizer (`WhitespaceTokenizer`):
                A `WhitespaceTokenizer` instance for word piece tokenization.
        """

        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

        def __init__(self,
                    vocab_file: str,
                    do_lower_case: bool = True,
                    never_split: Iterable = None,
                    unk_token: str = "[UNK]",
                    sep_token: str = "[SEP]",
                    pad_token: str = "[PAD]",
                    cls_token: str = "[CLS]",
                    strip_accents: bool = None,
                    model_max_length: int = 512,
                    **kwargs):
            """Construct a WordLevelTokenizer."""
            kwargs["model_max_length"] = model_max_length
            super().__init__(
                do_lower_case=do_lower_case,
                never_split=never_split,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                strip_accents=strip_accents,
                **kwargs,
            )

            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.vocab = load_vocab(vocab_file)
            # insert special token
            for token in [unk_token, sep_token, pad_token, cls_token]:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            self.whitespace_tokenizer = WhitespaceTokenizer(vocab=self.vocab, do_lower_case=do_lower_case,
                                                            unk_token=self.unk_token)

        @property
        def do_lower_case(self):
            """Returns whether or not to lowercase the input when tokenizing."""
            return self.whitespace_tokenizer.do_lower_case

        @property
        def vocab_size(self):
            """Returns the length of the vocabulary"""
            return len(self.vocab)

        def get_vocab(self):
            """Returns the vocabulary in a dictionary."""
            return dict(self.vocab, **self.added_tokens_encoder)

        def _tokenize(self,
                    text: str):
            """Tokenizes the input text into tokens."""
            if self.do_lower_case:
                text = text.lower()
            split_tokens = self.whitespace_tokenizer.tokenize(text)
            return split_tokens

        def _convert_token_to_id(self,
                                token: str):
            """Converts a token (`str`) in an id using the vocab."""
            return self.vocab.get(token, self.vocab.get(self.unk_token))

        def _convert_id_to_token(self,
                                index: int):
            """Converts an index (`int`) in a token (`str`) using the vocab."""
            return self.ids_to_tokens.get(index, self.unk_token)

        def convert_tokens_to_string(self,
                                    tokens: str):
            """Converts a sequence of tokens (`str`) in a single string."""
            out_string = " ".join(tokens).replace(" ##", "").strip()
            return out_string

        def build_inputs_with_special_tokens(self,
                                            token_ids_0: List[int],
                                            token_ids_1: Optional[List[int]] = None) -> List[int]:
            """Builds model inputs from a sequence or a pair of sequence.
            Builds model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
            adding special tokens. A BERT sequence has the following format:

            - single sequence: `[CLS] X [SEP]`
            - pair of sequences: `[CLS] A [SEP] B [SEP]`

            Args:
                token_ids_0 (`List[int]`):
                    List of ids to which the special tokens will be added.
                token_ids_1 (`List[int]`, `optional`):
                    Optional second list of ids for sequence pairs.

            Returns:
                `List[int]`: List of [input ids](../glossary#input-ids) with the appropriate special tokens.
            """
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + token_ids_1 + sep

        def get_special_tokens_mask(self,
                                    token_ids_0: List[int],
                                    token_ids_1: Optional[List[int]] = None,
                                    already_has_special_tokens: bool = False) -> List[int]:
            """Retrieve sequence ids from a token list that has no special tokens added."""

            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )

            if token_ids_1 is not None:
                return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1]

        def create_token_type_ids_from_sequences(self,
                                                token_ids_0: List[int],
                                                token_ids_1: Optional[List[int]] = None) -> List[int]:
            """Create a mask from the two sequences passed to be used in a sequence-pair classification task."""
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            if token_ids_1 is None:
                return len(cls + token_ids_0 + sep) * [0]
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

        def save_vocabulary(self,
                            save_directory: str,
                            filename_prefix: Optional[str] = None) -> Tuple[str]:
            """Saves the vocabulary (copy original file) and special tokens file to a directory."""
            index = 0
            if os.path.isdir(save_directory):
                vocab_file = os.path.join(
                    save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
                )
            else:
                vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
            with open(vocab_file, "w", encoding="utf-8") as writer:
                for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    if index != token_index:
                        logger.warning(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = token_index
                    writer.write(token + "\n")
                    index += 1
            return (vocab_file,)

``WhitespaceTokenizer``
-----------------------

Tokenizes a piece of text into its word pieces by matching whether the token is in the vocabulary.

**Attributes:**

- ``vocab``: A dictionary indicates the correspondence between words and ids within the vocabulary.
- ``do_lower_case``: A boolean variable indicating Whether or not to lowercase the input when tokenizing.
- ``unk_token``: A string representing the unknown token.

.. code-block:: python

    class WhitespaceTokenizer(object):
        """A tokenizer to conduct word piece tokenization.

        Tokenizes a piece of text into its word pieces by matching whether the token is in the vocabulary.

        Attributes:
            vocab (`Dict[str, int]`):
                A dictionary indicates the correspondence between words and ids within the vocabulary.
            do_lower_case (`bool`):
                A boolean variable indicating Whether or not to lowercase the input when tokenizing.
            unk_token (`str`):
                A string representing the unknown token.
        """

        def __init__(self,
                    vocab: Dict[str, int],
                    do_lower_case: bool,
                    unk_token: str):
            """Constructs a `WhitespaceTokenizer`."""
            self.vocab = vocab
            self.do_lower_case = do_lower_case
            self.unk_token = unk_token

        def tokenize(self,
                    text: str) -> List[str]:
            """Tokenizes a piece of text into its word pieces."""

            output_tokens = []
            for token in whitespace_tokenize(text):
                if token in self.vocab:
                    output_tokens.append(token)
                else:
                    output_tokens.append(self.unk_token)
            return output_tokens

import os
import yaml 
import dataclasses

from pathlib import Path 
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, HfArgumentParser

from .utils import check_web_and_convert_path


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.

    Arguments pertaining to what data we are going to input our model for training and eval, such as the config file
    path, dataset name, and the path of the training, validation, and testing file. By using `HfArgumentParser`, we can
    turn this class into argparse arguments to be able to specify them on the command line.
    """
    config_file: str = field(
        default=None, 
        metadata={"help": "Config file path."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonl file containing the test data."}
    )
    train_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers for training data. (Only meaningful for EAE)"}
    )
    validation_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers for valid data. (Only meaningful for EAE)"}
    )
    test_pred_file: Optional[str] = field(
        default=None, metadata={
            "help": "A jsonl file containing the predicted event triggers test data. (Only meaningful for EAE)"}
    )
    template_file: Optional[str] = field(
        default=None, metadata={"help": "Path to template file. (Only meaningful for mrc paradigm)"}
    )
    golden_trigger: bool = field(
        default=False,
        metadata={"help":" Whether or not to use golden trigger for EAE"}
    )
    test_exists_labels: bool = field(
        default=False,
        metadata={"help": "Whether test dataset exists labels"}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_out_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total output sequence length after tokenization."
        }
    )
    type2id_path: str = field(
        default=None,
        metadata={"help": "Path to type2id file."}
    )
    role2id_path: str = field(
        default=None,
        metadata={"help": "Path to role2id file."}
    )
    prompt_file: str = field(
        default=None, 
        metadata={
            "help": "Path to prompt file."
        }
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={
            "help": "Whether return token type ids"
        }
    )
    truncate_seq2seq_output: bool = field(
        default=False,
        metadata={
            "help": "Used for Seq2Seq. Whether truncate output labels."
        }
    )
    truncate_in_batch: bool = field(
        default=False,
        metadata={
            "help": "whether truncate in batch. False only if mrc."
        }
    )
    language: str = field(
        default="English",
        metadata={"help": "Data language."}
    )
    split_infer: bool = field(
        default=True,
        metadata={
            "help": "Whether split large dataset for inference. False only if truncate_in_batch"
        }
    )
    split_infer_size: int = field(
        default=500,
        metadata={
            "help": "Sub-batch size for split inference"
        }
    )
    eae_eval_mode: str = field(
        default="default",
        metadata={
            "help": "Evaluation mode for EAE, one of [default, loose, strict]"
        }
    )
    mrc_template_id: int = field(
        default=0,
        metadata={
            "help": "Mrc template, 0: role_name, 1: role_name in [trigger], 2: guidelines, 3: guidelines in [trigger]"
        }
    )
    insert_marker: bool = field(
        default=True,
        metadata={
            "help": "whether insert marker"
        }
    )
    consider_event_type: bool = field(
        default=False,
        metadata={
            "help": "Consider event type as type ids"
        }
    )


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from, such as the model type, model
    path, checkpoint path, hidden size, and aggregation method.
    """
    model_type: str = field(
        metadata={"help": "Model type."}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size"}
    )
    head_type: str = field(
        default="linear",
        metadata={"help": "Head type"}
    )
    head_scale: int = field(
        default=1,
        metadata={"help": "Head scale for classification head"}
    )
    aggregation: str = field(
        default="cls",
        metadata={"help": "Aggregation method"}
    )
    paradigm: str = field(
        default="token_classification",
        metadata={
            "help": "Paradigm of the method. Selected in ['token_classification', 'sequence_labeling', 'seq2seq', and 'mrc']."
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    '''
    For tranditional model (CNN, LSTM).
    '''
    word_embedding_dim: int = field(
        default=300,
        metadata={
            "help": "Word embedding dimension for tranditional word vector."
        }
    )
    position_embedding_dim: int = field(
        default=20,
        metadata={
            "help": "Position embedding dimension for tranditional word vector."
        }
    )
    type_embedding_dim: int = field(
        default=5,
        metadata={
            "help": "Type embedding dimension for tranditional word vector."
        }
    )
    num_position_embeddings: int = field(
        default=512,
        metadata={
            "help": "Number of position embeddings."
        }
    )
    hidden_dropout_prob: float = field(
        default=0.5,
        metadata={
            "help": "Dropout rate"
        }
    )
    vocab_file: float = field(
        default=None,
        metadata={
            "help": "Path to vocab file."
        }
    )
    has_type_embeddings: bool = field(
        default=False,
        metadata={
            "help": "type embeddings"
        }
    )



@dataclass 
class TrainingArguments(TrainingArguments):
    """Arguments pertaining to the configurations in the training process.

    Arguments pertaining to the configurations in the training process, such as the random seed, task name,
    early stopping patience and threshold, and max length.
    """
    seed: int = field(
        default=42,
        metadata={"help": "seed"}
    )
    task_name: str = field(
        default="ED",
        metadata={
            "help": "Task type. Selected in ['ED', 'EAE']"
        }
    )
    do_ED_infer: bool = field(
        default=False, 
        metadata={"help": "Whether infer on all splits."}
    )
    early_stopping_patience: int = field(
        default=7,
        metadata={"help": "Patience for early stopping."}
    )
    early_stopping_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold for early stopping."}
    )
    generation_max_length: int = field(
        default=128, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    generation_num_beams: int = field(
        default=3, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    ignore_pad_token_for_loss: bool = field(
        default=False, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    predict_with_generate: bool = field(
        default=False, 
        metadata={
            "help": "The maximum output length for encoder-decoder architecture (BART, T5)."
        }
    )
    pipeline: bool = field(
        default=False,
        metadata={
            "help": "Model parallelism."
        }
    )


class ArgumentParser(HfArgumentParser):
    """Alternative helper method that does not use `argparse` at all.

    Alternative helper method that does not use `argparse` at all, parsing the pre-defined yaml file with arguments
    instead loading a json file and populating the dataclass types.
    """
    def parse_yaml_file(self, yaml_file: str):
        """Parses the pre-defined yaml file with arguments."""
        data = yaml.safe_load(Path(yaml_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs,)

    def from_pretrained(self, yaml_file_name_or_path: str):
        path = check_web_and_convert_path(yaml_file_name_or_path, 'args')
        return self.parse_yaml_file(os.path.join(path, 'args.yaml'))
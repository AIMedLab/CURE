import collections
import logging
import math
import os
import pickle
import sys

import torch
from dataclasses import dataclass, field
# import wandb
import numpy as np

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import load_dataset, load_metric
from typing import Optional, List, Dict, Any, Tuple

from tokenization import MyTokenizer
from model import BertForSequenceClassification,BertForMaskedLM,LogisticRegression
from configuration_utils import ModelConfig

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    vocab_file: Optional[str] = field(
        default=None, metadata={"help": "The vocabulary file (a text file)"}
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

    time_embedding: bool = field(
        default=False,
        metadata={
            "help": "Whether to use time_embedding"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: str = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )

    eval_data_file: List[str] = field(
        default=None,
        metadata={"help": "The input eval data file (a text file)."}
    )

    data_path: str = field(
        default=None,
        metadata={"help": "The input training data path (directory)."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    mask_prediction: bool = field(
        default=False,
        metadata={
            "help": "Whether to do mask prediction"
        },
    )

    outcome_prediction: bool = field(
        default=False,
        metadata={
            "help": "Whether to do outcome prediction"
        },
    )

    counterfactual_inference: bool = field(
        default=False,
        metadata={
            "help": "Whether to do counterfactual inference"
        },
    )
    baseline_window: int = field(
        default=90,
        metadata={
            "help": "baseline_windowe"
        },
    )
    fix_window_length: int = field(
        default=30,
        metadata={
            "help": "fix_window_length"
        },
    )

    outcome_label: str = field(
        default=None,
        metadata={
            "help": "outcome_label"
        },
    )

    outcome_cutoff: int = field(
        default=None,
        metadata={
            "help": "outcome_cutoff"
        },
    )


    balance_dataset: bool = field(
        default=False,
        metadata={
            "help": "balance_dataset"
        },
    )

    if_pehe: bool = field(
        default=False,
        metadata={
            "help": "if_pehe"
        },
    )

    compared_drug: str = field(
        default=None,
        metadata={
            "help": "compared_drug"
        },
    )

    target_drug: str = field(
        default=None,
        metadata={
            "help": "target_drug"
        },
    )
    clean_out: str = field(
        default=None,
        metadata={
            "help": "clean_out"
        },
    )

    n_runs: int = field(
        default=0,
        metadata={
            "help": "n_runs"
        },
    )

    training_set_fraction: float = field(
        default=1,
        metadata={
            "help": "training_set_fraction"
        },
    )

    linear_fine_tune: bool = field(
        default=False,
        metadata={
            "help": "linear_fine_tune"
        },
    )


@dataclass
class myDataCollator:
    mask_prediction: bool = False
    outcome_prediction: bool = False
    counterfactual_inference: bool = False
    mlm_probability: float = 0.15
    tokenizer: MyTokenizer = None

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        outcome_labels = []
        token_type_ids = []
        visit_time_ids = []
        physical_time_ids = []

        input_ids_cf = []
        treatment_labels = []
        for b in batch:
            input_ids.append(b['input_ids'])
            if 'input_ids_cf' in b:
                input_ids_cf.append(b['input_ids_cf'])
            # input_ids_wo_treatment.append(b['input_ids_wo_treatment'])
            attention_mask.append(b['attention_mask'])
            outcome_labels.append(b['outcome'])
            token_type_ids.append(b['token_type_ids'])
            visit_time_ids.append(b['visit_time_ids'])
            physical_time_ids.append(b['physical_time_ids'])

            treatment_labels.append(b['treatment_label'])

        input_ids = torch.tensor(input_ids,dtype=torch.long)

        attention_mask = torch.tensor(attention_mask,dtype=torch.long)
        outcome_labels = torch.tensor(outcome_labels,dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)
        visit_time_ids = torch.tensor(visit_time_ids, dtype=torch.long)
        physical_time_ids = torch.tensor(physical_time_ids, dtype=torch.long)
        treatment_labels = torch.tensor(treatment_labels, dtype=torch.long)


        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                 "visit_time_ids": visit_time_ids, "physical_time_ids": physical_time_ids}

        if self.counterfactual_inference:
            input_ids_cf = torch.tensor(input_ids_cf, dtype=torch.long)
            batch['input_ids_cf'] = input_ids_cf

        if self.mask_prediction:
            input_ids, mask_labels, token_type_ids = self.torch_mask_tokens(input_ids,token_type_ids)
            batch['input_ids'] =  input_ids
            batch['mask_labels'] = mask_labels
            batch['token_type_ids'] = token_type_ids
        if self.outcome_prediction:
            batch['outcome_labels'] = outcome_labels
            batch['treatment_labels'] = treatment_labels
            # batch['input_ids_wo_treatment'] = input_ids_wo_treatment

        return batch

    def torch_mask_tokens(self, inputs: Any, token_type_ids: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        #
        # # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        #
        # # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        # mask_token_id = self.tokenizer.vocab.get(self.tokenizer.mask_token)
        # inputs[indices_replaced] = mask_token_id
        # token_type_ids[indices_replaced] = self.tokenizer.convert_token_ids_to_token_type_ids(mask_token_id)

        return inputs, labels, token_type_ids


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #####
    compared_drug = data_args.compared_drug
    target_drug = data_args.target_drug
    data_args.train_data_file = [
        os.path.join(data_args.data_path, target_drug),
        os.path.join(data_args.data_path, compared_drug)]


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # Load dataset

    if data_args.eval_data_file:
        # data_files = {"train": data_args.train_data_file, "validation": data_args.eval_data_file}
        data_files = {"train": data_args.train_data_file, "validation": data_args.eval_data_file}
        raw_datasets = load_dataset('json', data_files=data_files, field="data",cache_dir=model_args.cache_dir)

    else:
        data_files = {"train": data_args.train_data_file}
        raw_datasets = load_dataset('json', data_files=data_files, field="data",cache_dir=model_args.cache_dir)
        if data_args.outcome_prediction:
            overlap_pids = [data['pid'] for data in raw_datasets['train']]
            overlap_pids = collections.Counter(overlap_pids)
            overlap_pids = [pid for pid in overlap_pids if overlap_pids[pid]>1]
            raw_datasets = raw_datasets.filter(lambda example: example['pid'] not in overlap_pids)
        if data_args.validation_split_percentage > 0:
            raw_datasets = raw_datasets['train'].train_test_split(test_size=data_args.validation_split_percentage/100)
            raw_datasets['validation'] = raw_datasets['test']

    myTokenizer = MyTokenizer(
        vocab_file=model_args.vocab_file,
        baseline_window=data_args.baseline_window,
        fix_window_length=data_args.fix_window_length,
        treatment_list=[compared_drug.split('.')[0], target_drug.split('.')[0]] if data_args.outcome_prediction else None)

    max_seq_length = data_args.max_seq_length
    outcome_label = data_args.outcome_label
    outcome_cutoff = data_args.outcome_cutoff
    def prepare_data(example):
        result = myTokenizer.encode(example, max_length=max_seq_length)
        outcomes = example['outcome']
        result['outcome'] =  list(outcomes.values())[0]
 
        return result

    tokenized_datasets = raw_datasets.map(prepare_data,
                                          batched=False,
                                          num_proc=16,
                                          load_from_cache_file=not data_args.overwrite_cache)

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        logits = logits.softmax(dim=-1)
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")


    def compute_metrics(eval_preds):
        from scipy.special import softmax
        from sklearn.metrics import roc_auc_score,f1_score, average_precision_score, cohen_kappa_score,roc_curve, auc
        logits, labels = eval_preds
        logits = softmax(logits,axis=-1)
        preds = logits.argmax(axis=-1)

        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]

        results = metric.compute(predictions=preds, references=labels)
        results['logits'] = logits
        results['auc'] = roc_auc_score(labels,logits[:,1])
        results['f1'] = f1_score(labels,preds)
        results['aupr'] = average_precision_score(labels, logits[:,1])
        results['cohen'] = cohen_kappa_score(labels, preds)

        return results

    config = ModelConfig(
        vocab_size=len(myTokenizer),
        type_vocab_size=len(myTokenizer.type),
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_visit_time_embeddings = data_args.baseline_window+1,
        max_physical_time_embeddings= data_args.baseline_window//data_args.fix_window_length+1,
        time_embedding = model_args.time_embedding
    )

    if not model_args.model_name_or_path:
        logger.info("Train model from scratch...")
        if data_args.mask_prediction:
            model = BertForMaskedLM(config)
        else:
            model = BertForSequenceClassification(config)
    else:
        logger.info("Loading Model from pretrained...")
        if data_args.mask_prediction:
            model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path)
        else:
            model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path)
            if data_args.linear_fine_tune:
                for name, param in model.named_parameters():
                    if 'classifier' not in name:  # classifier layer
                        param.requires_grad = False

    data_collator = myDataCollator(
        tokenizer=myTokenizer,
        mask_prediction=data_args.mask_prediction,
        outcome_prediction=data_args.outcome_prediction,
        counterfactual_inference=data_args.counterfactual_inference)

    training_args.remove_unused_columns = False
    if data_args.mask_prediction:
        training_args.label_names = ["mask_labels"]

    if data_args.outcome_prediction:
        training_args.label_names = ["outcome_labels"]

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        # tokenizer=myTokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        logits = metrics.pop("eval_logits")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if os.path.exists(data_args.clean_out):
            out = open(data_args.clean_out, 'a')
        else:
            out = open(data_args.clean_out, 'w')

        out.write('{},{}\n'.format(str(metrics["eval_auc"]),str(metrics["eval_aupr"])))

if __name__ == "__main__":
    main()
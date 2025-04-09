import argparse
from typing import Dict

from dataset import WNUTDataLoader
from augmentation import Augmentation

import sys
import datasets
import evaluate
import logging

import torch
import transformers
import numpy as np

from typing import Dict
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from transformers import (AutoTokenizer, AutoModelForTokenClassification, AutoConfig,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

from datasets import Dataset, concatenate_datasets

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--text_colum")
    parser.add_argument("--tags_column")
    parser.add_argument("--output_dir", default=None, required=False)
    parser.add_argument("--path_results", type=str, required=False)
    parser.add_argument("--augmentation", type=str, default="no",
                        choices=["reverse_letter_case",
                                 "delete_character",
                                 "swap_characters",
                                 "substitute_character",
                                 "all",
                                 "no"]
                        )
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--segment", type=str, default="span",
                        choices=["span", "windows", "outside"])
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--stratified_samples", action="store_true")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

class Configuration:
    def __init__(self, model_name_or_path, num_labels, label_to_id, id_to_label):
        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 num_labels=num_labels,
                                                 label2id=label_to_id,
                                                 id2label=id_to_label)

    def __call__(self):
        return self.config


class Tokenization:
    def __init__(self, model_name_or_path, max_length: int, label2id: Dict, text_column_name: str,
                 ner_tag_column_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, add_prefix_space=True)
        self.max_length = max_length
        self.label2id = label2id
        self.text_column_name = text_column_name
        self.ner_tag_column_name = ner_tag_column_name

    def __call__(self, features: Dataset):
        tokenized_features = features.map(self.tokenize_and_align_labels,
                                          batched=True,
                                          remove_columns=[self.text_column_name,
                                                          self.ner_tag_column_name]
                                          )
        return tokenized_features

    def tokenize_and_align_labels(self, examples):
        tokenized = dict(input_ids=[],
                         attention_mask=[],
                         ner_tags=[])
        tokenized_inputs = self.tokenizer(examples[self.text_column_name],
                                          max_length=self.max_length,
                                          truncation=True,
                                          is_split_into_words=True
                                          )
        labels = []
        for i, label in enumerate(examples[self.ner_tag_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    if type(label[word_idx]) is str:
                        label_ids.append(self.align_labels_to_ids(label[word_idx], label2id=self.label2id))
                    else:
                        label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @staticmethod
    def align_labels_to_ids(label, label2id):
        return label2id[label]


class ModelForTokenClassification:
    def __init__(self, model_name_or_path, config):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config)


class AutoTokenClassifier:
    def __init__(self, trainer: transformers.Trainer, tokenized_dataset: datasets.Dataset, id_to_label_map: Dict):
        self.trainer = trainer
        self.tokenized_dataset = tokenized_dataset
        self.id_to_label_map = id_to_label_map
        self.predictions, self.gold = self._predict()

    def predict(self):
        # predictions, gold = self._predict()
        result_metrics = self.compute_metrics(self.predictions, self.gold)
        print(result_metrics)
        return result_metrics

    def _predict(self):
        predictions, gold, _ = self.trainer.predict(self.tokenized_dataset)
        predictions = np.argmax(predictions, axis=2)
        return predictions, gold

    def compute_metrics(self, predictions: torch.Tensor, gold: torch.Tensor):
        evaluations = dict(precision=None,
                           recall=None,
                           f1=None
                           )
        label_list = list(self.id_to_label_map.values())
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, gold)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, gold)
        ]

        # Metrics
        evaluations["precision"] = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        evaluations["recall"] = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        evaluations["f1"] = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        return evaluations

metrics = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metrics.compute(predictions=true_predictions, references=true_labels)
    return {"eval_precision": results["overall_precision"],
            "eval_recall": results["overall_recall"],
            "eval_f1": results["overall_f1"],
            }

if __name__=="__main__":
    args = arguments()
    args.model_name_or_path = "FacebookAI/roberta-base"
    args.text_column = "tokens"
    args.tags_column = "ner_tags"
    args.augmentation = "reverse_letter_case"
    args.ratio = 0.1
    args.segment = "span",
    args.windows = 1
    args.stratified_samples= True
    args.iteration = 1
    args.do_test=False
    args.seed = 42

    raw_dataset, label2id, id2label = WNUTDataLoader().__call__()
    label_list = list(label2id.keys())

    raw_train = raw_dataset["train"]
    raw_val = raw_dataset["validation"]
    raw_test = raw_dataset["test"]
    num_labels = len(id2label)

    # Insert noise into dataset
    augmentation = Augmentation(raw_train, id2label, stratified_samples=args.stratified_samples, ratio=args.ratio)
    augmentation_samples = augmentation()
    if args.augmentation != "no":
        if args.augmentation != "all":
            augmented = [sample for sample in augmentation_samples[args.augmentation]]
        else:
            augmented = [sample for key in augmentation_samples.keys() for sample in augmentation_samples[key]]
        # Insert noise injected instances to existing dataset with concatenation
        aug_dataset = Dataset.from_list(augmented).cast(raw_train.features)
        train_dataset = concatenate_datasets([raw_train, aug_dataset])
    else:
        train_dataset = raw_train

    config = Configuration(args.model_name_or_path,
                           num_labels=num_labels,
                           label_to_id=label2id,
                           id_to_label=id2label
                           )
    config = config()

    # Prepare data
    tokenization = Tokenization(args.model_name_or_path,
                                max_length=512,
                                label2id=label2id,
                                text_column_name=args.text_column,
                                ner_tag_column_name=args.tags_column
                                )
    tokenized_train_dataset = tokenization(train_dataset)
    tokenized_val_dataset = tokenization(raw_val)
    if args.do_test:
        # Test Prediction
        tokenized_test_dataset = tokenization(raw_test)

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenization.tokenizer,
                                                       return_tensors="pt"
                                                       )
    # Training
    model_for_token_classification = ModelForTokenClassification(args.model_name_or_path, config=config)
    model = model_for_token_classification.model

    # Training Arguments
    training_args = TrainingArguments(output_dir=args.output_dir,
                                      learning_rate=3e-5,
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=32,
                                      num_train_epochs=10,
                                      weight_decay=1e-2,
                                      do_train=True,
                                      do_eval=True,
                                      do_predict=args.do_test,
                                      gradient_accumulation_steps=4,
                                      eval_strategy="epoch",
                                      logging_strategy="epoch",
                                      save_strategy="epoch",
                                      save_total_limit=1,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="eval_f1",
                                      seed=args.seed
                                      )
    # Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_train_dataset,
                      eval_dataset=tokenized_val_dataset,
                      tokenizer=tokenization.tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics,
                      )
    train_results = trainer.train()
    metrics = train_results.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


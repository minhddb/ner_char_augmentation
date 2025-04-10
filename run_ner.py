#!/usr/bin/env python3

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from typing import Dict
from itertools import product

from dataset import WikiANNDataLoader, GermEval14, GermanLER, WNUTDataLoader
from augmentation import Augmentation

import torch
import evaluate
import numpy as np

from seqeval.scheme import IOB2
from seqeval.metrics import precision_score, recall_score, f1_score

import datasets
from datasets import Dataset, concatenate_datasets

import transformers
from transformers import (AutoTokenizer, AutoModelForTokenClassification, AutoConfig,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", choices=["germeval14",
                                                  "germanler",
                                                  "wikiann_de",
                                                  "wikiann_en",
                                                  "wnut17"])
    parser.add_argument("--model_name")
    parser.add_argument("--output_dir", default=None, required=False)
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
                        choices=["span", "windows"])
    parser.add_argument("--stratified_samples", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    return parser.parse_args()

class Configuration:
    def __init__(self, model_name, num_labels, label_to_id, id_to_label):
        self.config = AutoConfig.from_pretrained(model_name,
                                                 num_labels=num_labels,
                                                 label2id=label_to_id,
                                                 id2label=id_to_label)

    def __call__(self):
        return self.config


class Tokenization:
    def __init__(self, model_name, max_length: int, label2id: Dict, text_column_name: str,
                 ner_tag_column_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
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

def model_for_token_classification(model_name, config):
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model_init

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
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(gold, torch.Tensor):
            gold = gold.detach().cpu().numpy()
        
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

def compute_metrics(p):
    metrics = evaluate.load("seqeval")
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
    results = metrics.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {"eval_precision": results["overall_precision"],
            "eval_recall": results["overall_recall"],
            "eval_f1": results["overall_f1"],
            }

def generate_data(data, tags_column, ratios, stratified_samples=True):
    for ratio in ratios:
        if ratio == 0:
            yield ratio, "no_augmentation", data
        else:
            augmentation = Augmentation(raw_train, id2label, tag_column=tags_column, stratified_samples=stratified_samples, ratio=ratio, iteration=1)
            augmentation_samples = augmentation()
            for aug in ["reverse_letter_case", "delete_character", "swap_characters", "substitute_character", "all"]:
                if aug != "all":
                    augmented = [sample for sample in augmentation_samples[aug]]
                else:
                    augmented = [sample for key in augmentation_samples.keys() for sample in augmentation_samples[key]]
                augmented_dataset = Dataset.from_list(augmented) 
                # Insert noise injected instances to existing dataset with concatenation
                if "id" not in raw_train.features.keys():
                    augmented_dataset = augmented_dataset.remove_columns("id")
                casted_augmented_dataset = augmented_dataset.cast(raw_train.features)
                yield ratio, aug, concatenate_datasets([data, casted_augmented_dataset ])
            
if __name__=="__main__":
    args = arguments()

    TEXT_COLUMN = "tokens"
    TAGS_COLUMN = "ner_tags"

    if args.hf_dataset == "wikiann_de":
        raw_dataset, label2id, id2label = WikiANNDataLoader().__call__(language="de")
        label_list = list(label2id.keys())
        TAGS_COLUMN = "tags"

    if args.hf_dataset == "germanler":
        raw_dataset, label2id, id2label = GermanLER().__call__()
        raw_dataset = raw_dataset.remove_columns("ner_coarse_tags") # Modify parameter if needed
        label_list = list(label2id.keys())

    if args.hf_dataset == "germeval14":
        raw_dataset, label2id, id2label = GermEval14().__call__()
        raw_dataset = raw_dataset.remove_columns(["source", "nested_ner_tags"])
        label_list = list(label2id.keys())
    
    if args.hf_dataset == "wnut17":
        raw_dataset, label2id, id2label = WNUTDataLoader().__call__()
        label_list = list(label2id.keys())

    raw_train = raw_dataset["train"]
    raw_val = raw_dataset["validation"]
    raw_test = raw_dataset["test"]
    num_labels = len(id2label)

    if args.stratified_samples:
        path_results = f"results/{args.hf_dataset}_{args.model_name.split("/")[1].lower()}_stratified_samples.tsv"    
        output_dir = f"trained_{args.model_name.split("/")[1].lower()}_stratified" 
    else:
        path_results = f"results/{args.hf_dataset}_{args.model_name.split("/")[1].lower()}_random_samples.tsv"    
        output_dir = f"trained_{args.model_name.split("/")[1].lower()}_model" 

    
    with open(path_results, "w", encoding="utf-8") as out_f:
        out_f.write("model_name\tseed\taug_strategy\tratio\titeration\ttrain_size\tprecision\trecall\tf1\n")
    
    sample_ratios = [0, 0.1, 0.2, 0.4, 0.6, 0.8]
    # ratios = [0]
    for ratio, aug, train_dataset in generate_data(raw_train, TAGS_COLUMN, sample_ratios, args.stratified_samples):
        for run in range(1):
            # set_seed(run)
            config = Configuration(args.model_name,
                                   num_labels=num_labels,
                                   label_to_id=label2id,
                                   id_to_label=id2label
                                   )
            config = config()
            # Prepare data
            tokenization = Tokenization(args.model_name,
                                        max_length=512,
                                        label2id=label2id,
                                        text_column_name=TEXT_COLUMN,
                                        ner_tag_column_name=TAGS_COLUMN
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
            # Model
            # model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)

            # Training Arguments
            training_args = TrainingArguments(output_dir=output_dir,
                                              learning_rate=3e-5,
                                              per_device_train_batch_size=8,
                                              per_device_eval_batch_size=32,
                                              num_train_epochs=5,
                                              do_train=True,
                                              do_eval=True,
                                              gradient_accumulation_steps=4,
                                              eval_strategy="epoch",
                                              logging_strategy="epoch",
                                              save_strategy="epoch",
                                              save_total_limit=1,
                                              load_best_model_at_end=True,
                                              metric_for_best_model="eval_f1",
                                              seed=run
                                              )

            # Trainer
            trainer = Trainer(# model=model,
                              args=training_args,
                              train_dataset=tokenized_train_dataset,
                              eval_dataset=tokenized_val_dataset,
                              processing_class=tokenization.tokenizer,
                              data_collator=data_collator,
                              compute_metrics=compute_metrics,
                              model_init=model_for_token_classification(args.model_name, config),
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

            if args.do_test:
                output_preds = []
                classifier = AutoTokenClassifier(trainer, tokenized_test_dataset, id2label)
                test_results = classifier.predict()

                # Write results to tsv file for further analysis
                with open(path_results, "a", encoding="utf-8") as out_f:
                    out_f.write(f"{args.model_name}\t{run}\t{aug}\t{ratio}\t{1}\t{len(train_dataset["tokens"])}\t{test_results['precision']}\t{test_results['recall']}\t{test_results['f1']}\n")
                print(f"Run: {aug} {ratio} {1} {run}\n")
                print(f"P: {test_results['precision']} R: {test_results['recall']} F1: {test_results['f1']}")
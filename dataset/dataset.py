from typing import Dict, List, Iterable
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from datasets import load_dataset

@dataclass
class GermanLER:
    """
    Load GermanLER dataset from HF
    """
    hf_dataset_name_or_path: str = "elenanereiss/german-ler"

    def __call__(self, ner_tags="ner_tags"):
        """
        :param ner_tags: Options: 'ner_tags' and 'ner_coarse_tags'
        """
        dataset = load_dataset(self.hf_dataset_name_or_path, trust_remote_code=True)
        label2id = {label: i for i, label in enumerate(dataset["train"].features[ner_tags].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

@dataclass
class GermEval14:
    """
    Load GermEval2024 dataset from HF
    """
    hf_dataset_name_or_path: str = "GermanEval/germeval_14"

    def __call__(self):
        dataset = load_dataset(self.hf_dataset_name_or_path, trust_remote_code=True)
        label2id = {label: i for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

@dataclass
class WNUTDataLoader:
    """
    Load wnut17 dataset from HF
    """
    hf_dataset_name_or_path: str = "leondz/wnut_17"

    def __call__(self):
        dataset = load_dataset(self.hf_dataset_name_or_path, trust_remote_code=True)
        label2id = {label: i for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

@dataclass
class WikiANNDataLoader:
    """
    Load wikiann from HF
    """
    hf_dataset_name_or_path:str = "tner/wikiann"

    def __call__(self, language="en"):
        dataset = load_dataset(self.hf_dataset_name_or_path, language, trust_remote_code=True)
        # Labels Ids mapping taken from TNER
        label2id = {"B-LOC": 0,
                    "B-ORG": 1,
                    "B-PER": 2,
                    "I-LOC": 3,
                    "I-ORG": 4,
                    "I-PER": 5,
                    "O": 6
                    }
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

class Data:
    def __init__(self, dataset: Dataset, tag_column: str, mapping_dict: Dict):
        self.dataset = dataset
        self.mapping_dict = mapping_dict
        self.tag_column = tag_column
        self.non_entity_id = list(self.mapping_dict.keys())[0]

    def sequence_generator(self):
        for sequence in self.convert_id_to_entity():
            yield sequence["tokens"], sequence["ner_tags"]

    def get_entity_sequences(self):
        """
        Return all sequences containing at least one entity as list.
        :param ner_tags: Name of entity column.
        """
        entity_sequences = []
        for sequence in self._yield_from(self.dataset):
            if not all(tag == self.non_entity_id for tag in sequence[self.tag_column]):
                ent_sequence = [sequence["tokens"], [self.mapping_dict[tag] for tag in sequence[self.tag_column]]]
                if ent_sequence not in entity_sequences:
                    entity_sequences.append(ent_sequence)
        return entity_sequences

    def convert_id_to_entity(self):
        for seq in self._yield_from(self.dataset):
            sequence = {"id": None, "tokens": [], "ner_tags": []}
            sequence.update({"id": seq["id"], "tokens": seq["tokens"],
                             "ner_tags": [self.mapping_dict[tag] for tag in seq[self.tag_column]]})
            yield sequence

    @staticmethod
    def _yield_from(iterable: Iterable):
        yield from iterable

if __name__=="__main__":
    wikiann = WikiANNDataLoader().__call__(language="de")
    wnut_17 = WNUTDataLoader().__call__()
    germeval = GermEval14().__call__()
    germanler = GermanLER().__call__()
    print()
from typing import Dict
import random

from dataset import Data, Mappings, SequenceSegmentation
from augmentation.character_level import SimpleCharacterBasedPerturbation

from datasets import Dataset

class Augmentation(Data):
    def __init__(self,
                 data: Dataset,
                 id2label: Dict,
                 tag_column: str="ner_tags",
                 ratio: float=0.1,
                 stratified_samples:bool=False,
                 iteration: int = 1,
                 segment: str="span",
                 windows: int = 1,
                 ):
        super().__init__(data, tag_column, id2label)
        self.label2id = {val:key for key, val in id2label.items()}
        self.tags_column=tag_column
        self.stratified_samples = stratified_samples
        self.iteration = iteration
        self.segment = segment
        self.windows = windows
        self.entity_sequences = [sequence for sequence in self.get_entity_sequences()]
        self.ratio = ratio
        self.mappings = Mappings(self.entity_sequences)
        self.entity_to_sequence_distribution_mapping = self.mappings.map_entities_to_sequence_distribution()
        self.entity_to_sequences_mapping = self.mappings.map_entities_to_sequence()
        self.entity_to_spans_mapping = self.mappings.map_entities_to_spans()
        self.augmented_samples = dict(reverse_letter_case=[],
                                      delete_character=[],
                                      swap_characters=[],
                                      substitute_character=[]
                                      )
    def __call__(self, **kwargs):
        samples = [sample for sample in self.get_samples()]
        print(20 * "*" + "Augmentation" + 20 * "*")
        print(f"Segment: {self.segment}")
        print(f"Number of samples: {len(samples)}")
        print(f"Number of augmentation rounds: {self.iteration}")
        print(f"Augmentation ratio: {self.ratio}")
        # Use seq_id to replace original tokens with perturbed ones
        seq_id = 0
        processed=[]
        for sequence, tags in self.sequence_generator():
            if [sequence, tags] in samples:
                if [sequence, tags] not in processed:
                    processed.append([sequence, tags])
                    current_iteration = 0
                    while current_iteration < self.iteration:
                        # Set p dynamically to get more diverse augmented data
                        p_upper_bound = min(self.ratio, 0.7) # TODO: maybe check if other upper bound is better?
                        p = random.uniform(0.05, p_upper_bound)
                        # Re-create augmentation object for each rule in order to get more diverse synthetic data
                        # reverse case
                        augmentation = SimpleCharacterBasedPerturbation(sequence, tags, p=p, segment=self.segment, windows_size=self.windows)
                        augmented = augmentation.reverse_letter_case()
                        if augmented != sequence:
                            self.augmented_samples["reverse_letter_case"].append({"id": str(seq_id), "tokens": augmented, self.tags_column: [self.label2id[tag] for tag in tags]})
                        # remove chars
                        augmentation = SimpleCharacterBasedPerturbation(sequence, tags, p=p, segment=self.segment, windows_size=self.windows)
                        augmented = augmentation.delete_character()
                        if augmented != sequence:
                            self.augmented_samples["delete_character"].append({"id": str(seq_id), "tokens": augmented, self.tags_column: [self.label2id[tag] for tag in tags]})
                        # swap two chars
                        augmentation = SimpleCharacterBasedPerturbation(sequence, tags, p=p, segment=self.segment, windows_size=self.windows)
                        augmented = augmentation.swap_characters()
                        if augmented != sequence:
                            self.augmented_samples["swap_characters"].append({"id": str(seq_id), "tokens": augmented, self.tags_column: [self.label2id[tag] for tag in tags]})
                        # replace chars
                        augmentation = SimpleCharacterBasedPerturbation(sequence, tags, p=p, segment=self.segment, windows_size=self.windows)
                        augmented = augmentation.substitute_character()
                        if augmented != sequence:
                            self.augmented_samples["substitute_character"].append({"id": str(seq_id), "tokens": augmented, self.tags_column: [self.label2id[tag] for tag in tags]})
                        current_iteration += 1
            seq_id += 1
        return self.augmented_samples

    def get_samples(self):
        """
        Sample training data for augmentation.
        """
        samples = []
        if self.stratified_samples:
            for entity, value in self.entity_to_sequence_distribution_mapping.items():
                n_entity_samples = round(self.ratio * value)
                # samples.extend(self.entity_to_sequences_mapping[entity][:n_entity_samples])
                n = 0
                while n < n_entity_samples:
                    sample = self.entity_to_sequences_mapping[entity][n]
                    # Skip a sample if it is already in samples list to avoid duplication
                    if sample not in samples:
                        samples.append(sample)
                    n+=1
        else:
            n_samples = round(self.ratio * len(self.entity_sequences))
            samples = [sample for sample in random.sample(self.entity_sequences, k=n_samples)]
        yield from samples
from dataset.segmentation import SequenceSegmentation
from typing import List

class Mappings:
    def __init__(self, inp_dataset: List[List[str]]):
        self.inp_dataset = inp_dataset

    def map_entities_to_sequence_distribution(self):
        """
        Return a sorted dictionary of entity and number of sentences containing the entity.
        """
        entities_to_sequences_mapping = self.map_entities_to_sequence()
        return {k: v for k, v in sorted(
            {entity: len(entities_to_sequences_mapping[entity]) for entity in entities_to_sequences_mapping}.items(),
            key=lambda v: v[1]
        )}

    def map_entities_to_sequence(self):
        """
        Map whole sequence to it's belonging entity class. One sequence can belong to multiple classes.
        """
        entities_to_sequences_mapping = {}
        for entry in self.inp_dataset:
            sequence = entry[0]
            tags_list = entry[1]
            for tokens_ids in SequenceSegmentation(sequence, tags_list).get_entity_tokens_ids():
                entity = tags_list[tokens_ids[0]].strip("B-")
                if entity not in entities_to_sequences_mapping:
                    entities_to_sequences_mapping.update({entity:[]})
                # Avoid duplications -> When the same entity occurs in the same sentence
                if entry not in entities_to_sequences_mapping[entity]:
                    entities_to_sequences_mapping[entity].append(entry)
        return entities_to_sequences_mapping

    def map_entities_to_spans(self):
        """
        Return a mapping dictionary with entity as key and list of lists of spans as values.
        """
        entities_to_spans_mapping = {}
        c = 0
        for entry in self.inp_dataset:
            sequence = entry[0]
            tags_list = entry[1]
            for tokens_ids in SequenceSegmentation(sequence, tags_list).get_entity_tokens_ids():
                entity = tags_list[tokens_ids[0]].strip("B-")
                span = [sequence[i] for i in tokens_ids]
                if entity not in entities_to_spans_mapping:
                    entities_to_spans_mapping.update({entity: []})
                    # entities_to_spans_mapping[entity].append(span)
                #if span not in entities_to_spans_mapping[entity]:
                if entity == "location" and span in entities_to_spans_mapping[entity]:
                    c += 1
                else:
                    entities_to_spans_mapping[entity].append(span)
        return entities_to_spans_mapping

    def map_tags_to_tokens(self):
        """
        Create labels to tokens mapping. This mapping will be utilised for label-wise and similarity-wise replacements.
        :return:
        """
        labels_to_tokens_mappings = {}
        for entry in self.inp_dataset:
            for token, tag in zip(entry[0], entry[1]):
                if tag not in labels_to_tokens_mappings.keys():
                    labels_to_tokens_mappings.update({tag: [token]})
                else:
                    if token not in labels_to_tokens_mappings[tag]:
                        labels_to_tokens_mappings[tag].append(token)
        return labels_to_tokens_mappings


import random
import string
import itertools
from typing import List

import numpy as np
from dataset import SequenceSegmentation

class SimpleCharacterBasedPerturbation(SequenceSegmentation):
    def __init__(self, sequence: List[str], tags: List[str], p: float=0.1, **kwargs):
        super().__init__(sequence=sequence, tags=tags)
        self.p = p # p determines the number augmentation candidates
        # Get list of segment tokens ids
        self.segments_tokens_ids = list(itertools.chain.from_iterable(self.__call__(**kwargs)))
        self.candidates = self.candidates()

    def reverse_letter_case(self):
        """
        Randomly reversing the case of a letter. I.e.: lower -> upper  and upper -> lower
        :return: List of tokens containing reversed letters cases
        """
        sequence = []
        for token in self.sequence:
            if self.sequence.index(token) in self.candidates:
                # Select candidate characters for augmentation
                chars = list(token)
                n_candidates = round(self.p * len(token))
                if n_candidates == 0: n_candidates = 1 # enforce at least 1 candidate to augment
                char_candidates = np.random.choice([i for i in range(len(chars))], n_candidates, replace=False)
                for c in char_candidates:
                    chars[c] = chars[c].upper() if chars[c].islower() else chars[c].lower()
                token = "".join(chars)
            sequence.append(token)
        assert len(sequence) == len(self.sequence)
        return sequence

    def delete_character(self):
        """
        Delete a character / letter from tokens.
        """
        sequence = []
        for token in self.sequence:
            if self.sequence.index(token) in self.candidates:
                # Length of current token should be at least 2
                if len(token) >= 2:
                    # Determine number of candidate characters to remove
                    chars = list(token)
                    n_candidates = round(self.p * len(token))
                    if n_candidates == 0: n_candidates = 1
                    candidates = np.random.choice(chars, n_candidates, replace=False).tolist()
                    for c in candidates:
                        chars.remove(c)
                    token = "".join(chars)
            sequence.append(token)
        assert len(sequence) == len(self.sequence)
        return sequence

    def swap_characters(self):
        """ Swap position of two random characters that are not the first or the last within the token.
        E.g.: 'Letter' -> 'Leettr'
        """
        sequence = []
        for token in self.sequence:
            if self.sequence.index(token) in self.candidates:
                # Current token should contain at least 4 characters
                if len(token) >= 4 :
                    chars = list(token)
                    # Get character ids for augmentation but ignore the first and the last ones
                    chars_positions = [i for i,_ in enumerate(token)]
                    candidates = np.random.choice(chars_positions[1:-1], 2, replace=False).tolist()
                    tmp_char = chars[candidates[0]]
                    chars[candidates[0]] = token[candidates[1]]
                    chars[candidates[1]] = tmp_char
                    token = "".join(chars)
            sequence.append(token)
        assert len(sequence) == len(self.sequence)
        return sequence

    def substitute_character(self):
        """ Substitute a random character from token with another one of the
        same type.
        I.e. substitute an upper case vowel with another one.
        E.g.: 'A' -> 'E', '1' -> 9.
        """
        sequence = []
        for token in self.sequence:
            if self.sequence.index(token) in self.candidates:
                chars = list(token)
                # Select character to substitute
                char_candidate = np.random.choice([i for i,_ in enumerate(token)], 1).tolist()[0]
                if chars[char_candidate].isalnum():
                    if chars[char_candidate].isdigit():
                        chars[char_candidate] = random.choice(string.digits)
                    else:
                        char = token[char_candidate]
                        replacement = ""
                        while char == replacement:
                            replacement = random.choice(string.ascii_lowercase)
                        if char.isupper:
                            replacement = replacement.upper()
                        chars[char_candidate] = replacement
                token = "".join(chars)
            sequence.append(token)
        assert len(sequence) == len(self.sequence)
        return sequence


    def candidates(self):
        """ Determine the number of tokens for augmentation by taking the
        product of p, probability value and number of all segment tokens.
        E.g.: p: 0.3, len(tokens_ids): 10 -> 3 candidates.
        """
        number_of_candidates = round(self.p * len(self.segments_tokens_ids))
        # We need to extract at least 1 candidate
        if number_of_candidates == 0:
            number_of_candidates = 1
        return np.random.choice(self.segments_tokens_ids, number_of_candidates, replace=False).tolist()

if __name__ == "__main__":
    tokens = ["My", "name", "is", "Monkey", "D.", "Luffy", ".", "I", "'", "m", "gonna", "be", "King", "of", "the", "Pirates"]
    tags = ["O", "O", "O", "B-name", "I-name", "I-name", "O", "O", "O", "O", "O", "O", "B-title", "I-title", "I-title", "I-title"]
    char_aug = SimpleCharacterBasedPerturbation(tokens, tags)
    print(char_aug.reverse_letter_case())
    print(char_aug.delete_character())
    print(char_aug.swap_characters())
    print(char_aug.substitute_character())
    print()


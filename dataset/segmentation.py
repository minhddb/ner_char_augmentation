from typing import List


class SequenceSegmentation:
    def __init__(self, sequence: List[str], tags: List[str]):
        self.sequence = sequence
        self.tags = tags

    def __call__(self, segment: str = "span", windows_size: int = 1):
        """

        :param segment: Indicate which segment should be yielded. Options: 'span', 'windows' and 'all'. Default: 'span'.
        :return:
        """
        list_of_tokens_ids = []
        if segment == "span":
            for tokens_ids in self.get_entity_tokens_ids():
                list_of_tokens_ids.append(tokens_ids)
        if segment == "windows":
            for tokens_ids in self.get_context_windows_tokens_ids(windows_size=windows_size):
                list_of_tokens_ids.append(tokens_ids)
        if segment == "all":
            return self.get_all_split_segments()
        if list_of_tokens_ids:
            return list_of_tokens_ids

    def get_all_split_segments(self):
        """
        Split tokens sequence into different segments based on corresponding tags.
        E.g.: ["O", "B", "I", "O", "O", "O", "B", "I", "I", "O"]
        -> [[0], [1, 2], [3, 4, 5], [6, 7, 8], [9]]
        :return: List of split tokens segments
        """
        segments = []
        segment = []
        for i, _ in enumerate(self.tags):
            try:
                if self.tags[i] == "O":
                    segment.append(i)
                    if self.tags[i + 1].startswith("B-"):
                        segments.append(segment)
                        segment = []
                else:
                    segment.append(i)
                    if ((self.tags[i].startswith("I-")
                         or self.tags[i].startswith("B-"))
                            and not self.tags[i + 1].startswith("I-")):
                        segments.append(segment)
                        segment = []
            except IndexError:
                segments.append(segment)
        return segments

    def get_entity_tokens_ids(self):
        """
        Yield annotated spans and their positions within the span for further processing.
        :return: List of segme  nt tokens and list of position ids.
        E.g.: ["B-name", "I-name", "O", "O", "O"] --> [0, 1]
        """
        tokens_ids = []
        for i, _ in enumerate(self.tags):
            try:
                if (self.tags[i].startswith("B-")
                        or self.tags[i].startswith("I-")):
                    tokens_ids.append(i)
                    if (self.tags[i + 1] == "O"
                            or self.tags[i + 1].startswith("B-")):
                        yield tokens_ids
                        tokens_ids = []
            except IndexError:
                yield tokens_ids
                tokens_ids = []

    def get_context_windows_tokens_ids(self, windows_size: int = 1):
        for entity_tokens_ids in self.get_entity_tokens_ids():
            list_of_tokens_ids = self._get_context_windows(token_ids=entity_tokens_ids,
                                                           n_tokens=len(self.sequence),
                                                           windows_size=windows_size
                                                           )

            for tokens_ids in list_of_tokens_ids:
                yield tokens_ids

    @staticmethod
    def _get_context_windows(token_ids: List, n_tokens: int, windows_size: int = 1):
        """
        Extract left and right context from sequence
        :param token_ids: Input token ids as list
        :param n_tokens: Number of tokens
        :param windows_size: Context window size
        """
        l_context = [i for i in range(token_ids[0] - windows_size, token_ids[0]) if i >= 0]
        r_context = [i for i in range(token_ids[-1] + 1, token_ids[-1] + windows_size + 1) if i <= n_tokens - 1]
        return [l_context, r_context]

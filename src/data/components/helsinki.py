import re
import os

from src.utils.text_processing import (
    python_lowercase,
    python_remove_punctuation,
    python_lowercase_remove_punctuation,
)


def process_file(filename):
    """
    Processes a helsinki file and returns a list of utterances.
    Output format is a list of samples that look like:
        {
        'filename': '1034_121119_000001_000001.txt',
        'word_labels': [
            {'text': 'The',
            'discrete_prominence': 0,
            'discrete_word_boundary': 0,
            'real_prominence': 0.171,
            'real_word_boundary': 0.0},
            {'text': 'Law',
            'discrete_prominence': 2,
            'discrete_word_boundary': 2,
            'real_prominence': 1.826,
            'real_word_boundary': 2.0}],
            'text': 'The Law '
            }
        ]
    """
    utterances = []
    current_utterance = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("<file>"):
                if current_utterance is not None:
                    utterances.append(current_utterance)
                current_utterance = {
                    "filename": line.split("\t")[1],
                    "word_labels": [],
                    "text": "",
                }
            else:
                match = re.match(r"^(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)$", line)
                if match:
                    word = {
                        "text": match.group(1),
                        "discrete_prominence": int(match.group(2))
                        if match.group(2) != "NA"
                        else None,
                        "discrete_word_boundary": int(match.group(3))
                        if match.group(3) != "NA"
                        else None,
                        "real_prominence": float(match.group(4))
                        if match.group(4) != "NA"
                        else None,
                        "real_word_boundary": float(match.group(5))
                        if match.group(5) != "NA"
                        else None,
                    }
                    current_utterance["word_labels"].append(word)
                    current_utterance["text"] += match.group(1) + " "

    if current_utterance is not None:
        utterances.append(current_utterance)

    return utterances


class HelsinkiProminenceExtractor:
    """
    Extract and access the prominence features of the Helsinki corpus.
    """

    def __init__(
        self,
        root_dir,
        filename,
        lab_dir=None,
        lowercase=False,
        remove_punctuation=False,
    ):
        """
        filename: path to the file containing the prominence labels,
        """
        self.root_dir = root_dir
        self.lab_dir = lab_dir
        self.filename = filename
        self.utterances = process_file(os.path.join(root_dir, filename))
        # print("Loaded {} utterances".format(len(self.utterances)))
        self.length = len(self.utterances)
        self.filename_to_index = {self.get_filename(i): i for i in range(self.length)}
        self.index_to_filename = {i: self.get_filename(i) for i in range(self.length)}

        # Preprocessing
        if lowercase and remove_punctuation:
            self.preprocess_fct = python_lowercase_remove_punctuation
        elif lowercase:
            self.preprocess_fct = python_lowercase
        elif remove_punctuation:
            self.preprocess_fct = python_remove_punctuation
        else:
            self.preprocess_fct = lambda x: x

    def __getitem__(self, index):
        return self.utterances[index]

    def __len__(self):
        return self.length

    def get_filepath(self, index):
        # TODO: not yet implemented
        raise NotImplementedError
        filename = self.get_filename(index)
        print(f"filename: {filename}")
        reader, book, ut1, ut2 = filename.split("_")
        ut = ut1 + "_" + ut2
        print(f"reader: {reader}, book: {book}, ut: {ut}")

    def get_index(self, filename):
        return self.filename_to_index[filename]

    def get_filename(self, index):
        return self.utterances[index]["filename"].replace(".txt", "")

    def get_text(self, index):
        text = ""
        for word in self.utterances[index]["word_labels"]:
            text += word["text"] + " "
        return self.preprocess_fct(text[:-1])  # remove last space

    def get_word_labels(self, index):
        return self.utterances[index]["word_labels"]

    def get_discrete_prominence(self, index, classes=3, min_length=0):
        if classes == 3:
            return [
                word["discrete_prominence"]
                for word in self.utterances[index]["word_labels"]
                if len(word["text"]) >= min_length
            ]
        elif classes == 2:
            return [
                1 if word["discrete_prominence"] > 0 else 0
                for word in self.utterances[index]["word_labels"]
                if len(word["text"]) >= min_length
            ]

    def get_real_prominence(self, index, min_length=0):
        return [
            word["real_prominence"]
            for word in self.utterances[index]["word_labels"]
            if len(word["text"]) >= min_length
        ]

    def get_discrete_word_boundary(self, index, classes=3, min_length=0):
        if classes == 3:
            return [
                word["discrete_word_boundary"]
                for word in self.utterances[index]["word_labels"]
                if len(word["text"]) >= min_length
            ]
        elif classes == 2:
            return [
                1 if word["discrete_word_boundary"] > 0 else 0
                for word in self.utterances[index]["word_labels"]
                if len(word["text"]) >= min_length
            ]

    def get_real_word_boundary(self, index, min_length=0):
        return [
            word["real_word_boundary"]
            for word in self.utterances[index]["word_labels"]
            if len(word["text"]) >= min_length
        ]

    def get_all_texts(self, min_length=0):
        return [
            self.get_text(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_discrete_prominence(self, classes=3, min_length=0):
        return [
            self.get_discrete_prominence(i, classes)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_real_prominence(self, min_length=0):
        return [
            self.get_real_prominence(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_discrete_word_boundary(self, classes=3, min_length=0):
        return [
            self.get_discrete_word_boundary(i, classes)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_all_real_word_boundary(self, min_length=0):
        return [
            self.get_real_word_boundary(i)
            for i in range(self.length)
            if len(self.get_text(i)) >= min_length
        ]

    def get_max_text_length_in_words(self):
        return max([len(self.get_text(i).split(" ")) for i in range(self.length)])

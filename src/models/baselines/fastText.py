import fasttext
import fasttext.util as ft_util


class FastTextModel:
    def __init__(self, model_path: str):
        """
        Initialize FastTextModel with path to .bin file.

        Parameters:
        model_path (str): Path to the pre-trained FastText model (.bin file).
        """
        self.model = fasttext.load_model(model_path)

    def get_word_embedding(self, word: str):
        """
        Returns the FastText vector for a given word.

        Parameters:
        word (str): Word to get FastText vector for.

        Returns:
        numpy.ndarray: FastText vector for the given word.
        """
        return self.model.get_word_vector(word)

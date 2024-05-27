import numpy as np
import os
import pickle
import bisect

from scipy.stats import pearsonr


class LSTMEncodingAnalyzer:
    def __init__(self, dir_path, batch_analysis=False):
        """
        ::param dir_path: path to the directory containing the analysis files
        ::param batch_analysis: if True, the analysis is done per batch, otherwise it is done over the whole dataset flattened
        """
        self.dir_path = dir_path
        self.predictions = []
        self.latents = []
        self.loss_masks = []
        self.targets = []

        self.batch_analysis = batch_analysis
        self.cumulative_sum = None  # only for per batch analysis

        self._load_data()

    def _load_data(self):
        files = os.listdir(self.dir_path)
        for file in sorted(files):
            file_path = os.path.join(self.dir_path, file)
            if file.startswith("loss_masks"):
                self.loss_masks.append(np.load(file_path))
            elif file.startswith("predictions"):
                self.predictions.append(np.load(file_path))
            elif file.startswith("targets"):
                self.targets.append(np.load(file_path))
            elif file.startswith("latents"):
                self.latents.append(np.load(file_path))
            else:
                print(f"Weird file: {file_path}")

        if not self.batch_analysis:
            self.loss_masks = [item for sublist in self.loss_masks for item in sublist]
            self.predictions = [
                item for sublist in self.predictions for item in sublist
            ]
            self.targets = [item for sublist in self.targets for item in sublist]
            self.latents = [item for sublist in self.latents for item in sublist]
        else:
            self.cumulative_sum = np.cumsum([batch.shape[0] for batch in self.targets])

        print(
            f"all lengths {len(self.loss_masks)} {len(self.predictions)} {len(self.targets)} {len(self.latents)}"
        )

    def __len__(self):
        return len(self.targets)

    def _get_element_by_index(self, data_list, index):
        return data_list[index]

    def get_latent(self, index):
        return self._get_element_by_index(self.latents, index)

    def get_all_latents(self):
        if not self.batch_analysis:
            return self.latents
        else:
            return np.concatenate(self.latents, axis=0)

    def get_all_predictions(self):
        if not self.batch_analysis:
            return self.predictions
        else:
            return np.concatenate(self.predictions, axis=0)

    def get_all_targets(self):
        if not self.batch_analysis:
            return self.targets
        else:
            return np.concatenate(self.targets, axis=0)

    def get_all_loss_masks(self):
        if not self.batch_analysis:
            return self.loss_masks
        else:
            return np.concatenate(self.loss_masks, axis=0)

    def get_prediction(self, index):
        return self._get_element_by_index(self.predictions, index)

    def get_target(self, index):
        return self._get_element_by_index(self.targets, index)

    def get_loss_mask(self, index):
        return self._get_element_by_index(self.loss_masks, index)

    def compute_mse(self, index, per_element=True):
        """
        ::param index: index of the batch to compute the mse for
        ::param per_element: if True, the mse is computed per element of the batch, otherwise it is computed over the whole batch
        That means if self.batch_analysis and per_element==True, then the result will be the same as without batch analysis
        """
        if not self.batch_analysis or not per_element:
            target = self.get_target(index)
            prediction = self.get_prediction(index)
            loss_mask = self.get_loss_mask(index)

            squared_errors = (target - prediction) ** 2
            masked_squared_errors = squared_errors[loss_mask == 1]

            mse = np.mean(masked_squared_errors)
            return mse
        else:
            # compute mse per element of the batch
            errors = []
            for i in range(len(self)):
                target = self.get_target(i)
                prediction = self.get_prediction(i)
                loss_mask = self.get_loss_mask(i)
                for j in range(len(target)):
                    squared_errors = (target[j] - prediction[j]) ** 2
                    masked_squared_errors = squared_errors[loss_mask[j] == 1]
                    mse = np.mean(masked_squared_errors)
                    errors.append(mse)
            return np.mean(errors)

    def compute_mse_per_element(self, index):
        return self.compute_mse(index, per_element=True)

    def compute_all_mse(self, per_element=True):
        mses = []
        for i in range(len(self)):
            mse = self.compute_mse(i, per_element=per_element)
            mses.append(mse)
            if np.isnan(mse):
                print(
                    f"mse is nan for index {i}, predictions {self.predictions[i][self.loss_masks == 1]}, targets {self.targets[i][self.loss_masks == 1]}, mask {self.loss_masks[i][self.loss_masks == 1]}"
                )
        # return the non-nan mean
        return np.nanmean(mses)

    def compute_pearson(self, index, per_element=True):
        """
        ::param index: index of the batch to compute the mse for
        ::param per_element: if True, the mse is computed per element of the batch, otherwise it is computed over the whole batch
        That means if self.batch_analysis and per_element==True, then the result will be the same as without batch analysis
        """
        if not self.batch_analysis or not per_element:
            target = self.get_target(index)
            prediction = self.get_prediction(index)
            loss_mask = self.get_loss_mask(index)

            pearson = pearsonr(target, prediction)[0]
            return pearson
        else:
            # compute mse per element of the batch
            pearsons = []
            for i in range(len(self)):
                target = self.get_target(i)
                prediction = self.get_prediction(i)
                loss_mask = self.get_loss_mask(i)
                for j in range(len(target)):
                    pearson = pearsonr(target[j], prediction[j])[0]
                    pearsons.append(pearson)
            return np.mean(pearsons)

    def compute_pearson_per_element(self, index):
        return self.compute_pearson(index, per_element=True)


class LanguageModelAnalyzer:
    def __init__(self, dir_path, batch_analysis=False):
        """
        ::param dir_path: path to the directory containing the analysis files
        ::param batch_analysis: if True, the analysis is done per batch, otherwise it is done over the whole dataset flattened
        """
        self.dir_path = dir_path
        self.attention_masks = []
        self.input_ids = []
        self.input_texts = []
        self.labels = []
        self.loss_masks = []
        self.original_labels = []
        self.preds = []
        self.word_to_tokens = []

        self.batch_analysis = batch_analysis
        self.cumulative_sum = None  # only for per batch analysis

        self._load_data()

    def _load_data(self):
        files = os.listdir(self.dir_path)
        for file in sorted(files):
            file_path = os.path.join(self.dir_path, file)
            if file.startswith("test_attention_mask"):
                self.attention_masks.append(np.load(file_path))
            elif file.startswith("test_input_ids"):
                self.input_ids.append(np.load(file_path))
            elif file.startswith("test_input_text"):
                with open(file_path, "rb") as f:
                    self.input_texts.append(pickle.load(f))
            elif file.startswith("test_labels"):
                self.labels.append(np.load(file_path))
            elif file.startswith("test_loss_mask"):
                self.loss_masks.append(np.load(file_path))
            elif file.startswith("test_original_labels"):
                with open(file_path, "rb") as f:
                    self.original_labels.append(pickle.load(f))
            elif file.startswith("test_preds"):
                self.preds.append(np.load(file_path))
            elif file.startswith("test_word_to_tokens"):
                with open(file_path, "rb") as f:
                    self.word_to_tokens.append(pickle.load(f))

        if not self.batch_analysis:
            self.attention_masks = [
                item for sublist in self.attention_masks for item in sublist
            ]
            self.input_ids = [item for sublist in self.input_ids for item in sublist]
            self.input_texts = [
                item for sublist in self.input_texts for item in sublist
            ]
            self.labels = [item for sublist in self.labels for item in sublist]
            self.loss_masks = [item for sublist in self.loss_masks for item in sublist]
            self.original_labels = [
                item for sublist in self.original_labels for item in sublist
            ]
            self.preds = [item for sublist in self.preds for item in sublist]
            self.word_to_tokens = [
                item for sublist in self.word_to_tokens for item in sublist
            ]
        else:
            self.cumulative_sum = np.cumsum(
                [batch.shape[0] for batch in self.input_ids]
            )

        print(
            f"all lengths {len(self.input_ids)} {len(self.input_texts)} {len(self.labels)} {len(self.original_labels)} {len(self.preds)} {len(self.word_to_tokens)}"
        )

    def __len__(self):
        return len(self.input_ids)

    def _get_element_by_index(self, data_list, index):
        return data_list[index]

    def get_attention_mask(self, index):
        return self._get_element_by_index(self.attention_masks, index)

    def get_input_id(self, index):
        return self._get_element_by_index(self.input_ids, index)

    def get_input_text(self, index):
        return self.input_texts[index]

    def get_label(self, index):
        return self._get_element_by_index(self.labels, index)

    def get_loss_mask(self, index):
        return self._get_element_by_index(self.loss_masks, index)

    def get_original_label(self, index):
        return self.original_labels[index]

    def get_pred(self, index):
        return self._get_element_by_index(self.preds, index)

    def get_word_to_token(self, index):
        return self.word_to_tokens[index]

    def compute_mse(self, index, per_element=True):
        """
        ::param index: index of the batch to compute the mse for
        ::param per_element: if True, the mse is computed per element of the batch, otherwise it is computed over the whole batch
        That means if self.batch_analysis and per_element==True, then the result will be the same as without batch analysis
        """
        if not self.batch_analysis or not per_element:
            label = self.get_label(index)
            pred = self.get_pred(index)
            loss_mask = self.get_loss_mask(index)

            squared_errors = (label - pred) ** 2
            masked_squared_errors = squared_errors[loss_mask == 1]

            mse = np.mean(masked_squared_errors)
            return mse
        else:
            # compute mse per element of the batch
            errors = []
            for i in range(len(self)):
                label = self.get_label(i)
                pred = self.get_pred(i)
                loss_mask = self.get_loss_mask(i)
                for j in range(len(label)):
                    squared_errors = (label[j] - pred[j]) ** 2
                    masked_squared_errors = squared_errors[loss_mask[j] == 1]
                    mse = np.mean(masked_squared_errors)
                    errors.append(mse)
            return np.mean(errors)

    def compute_sentence_mse_unreduced(self, index):
        label = self.get_label(index)
        pred = self.get_pred(index)
        loss_mask = self.get_loss_mask(index)

        squared_errors = (label - pred) ** 2
        masked_squared_errors = squared_errors[loss_mask == 1]

        return masked_squared_errors

    def compute_mse_over_dataset(self, per_element=True):
        mse_list = []
        for i in range(len(self)):
            mse_list.append(self.compute_mse(i, per_element=per_element))
        return np.mean(mse_list)

    def compute_stats_per_word(self):
        word_mse = {}
        word_count = {}

        for i in range(len(self.input_texts)):
            words = self.input_texts[i].split(" ")
            labels = self.original_labels[i]
            preds = self.get_pred(i)[self.get_loss_mask(i) == 1]
            # print(f"words, labels, preds {words} {labels} {preds}")
            for j, word in enumerate(words):
                # print(f"labels[j], preds[j] {labels[j]} {preds[j]}")
                mse = (labels[j] - preds[j]) ** 2
                if word in word_mse:
                    word_mse[word].append(mse)
                    word_count[word] += 1
                else:
                    word_mse[word] = [mse]
                    word_count[word] = 1

        mean_std_mse_per_word = {
            word: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": word_count[word],
            }
            for word, mse_list in word_mse.items()
        }
        return mean_std_mse_per_word

    def compute_stats_per_sentence_length(self):
        sentence_length_mse = {}
        sentence_length_count = {}

        for i in range(len(self.input_texts)):
            sentence_length = len(self.input_texts[i].split(" "))
            mse = self.compute_mse(i)

            if sentence_length in sentence_length_mse:
                sentence_length_mse[sentence_length].append(mse)
                sentence_length_count[sentence_length] += 1
            else:
                sentence_length_mse[sentence_length] = [mse]
                sentence_length_count[sentence_length] = 1

        mean_std_mse_per_sentence_length = {
            length: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": sentence_length_count[length],
            }
            for length, mse_list in sentence_length_mse.items()
        }
        return mean_std_mse_per_sentence_length

    def compute_stats_per_word_position(self):
        word_position_mse = {}
        word_position_count = {}

        for i in range(len(self.input_texts)):
            words = self.input_texts[i].split(" ")
            labels = self.original_labels[i]
            preds = self.get_pred(i)[self.get_loss_mask(i) == 1]

            for j, (label, pred) in enumerate(zip(labels, preds)):
                mse = (label - pred) ** 2
                if j in word_position_mse:
                    word_position_mse[j].append(mse)
                    word_position_count[j] += 1
                else:
                    word_position_mse[j] = [mse]
                    word_position_count[j] = 1

        mean_std_mse_per_word_position = {
            position: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": word_position_count[position],
            }
            for position, mse_list in word_position_mse.items()
        }
        return mean_std_mse_per_word_position


class BaselineAnalyzer:
    def __init__(self, dir_path):
        """
        :param dir_path: path to the directory containing the analysis files
        """
        self.dir_path = dir_path
        self.input_texts = []
        self.labels = []
        self.preds = []

        self.__load_data()

    def __load_data(self):
        files = os.listdir(self.dir_path)
        for file in sorted(files):
            file_path = os.path.join(self.dir_path, file)
            if file.startswith("labels"):
                with open(file_path, "rb") as f:
                    self.labels = pickle.load(f)
            elif file.startswith("pred"):
                with open(file_path, "rb") as f:
                    self.preds = pickle.load(f)
            elif file.startswith("texts"):
                with open(file_path, "rb") as f:
                    self.input_texts = pickle.load(f)

        print(
            f"all lengths {len(self.input_texts)} {len(self.labels)} {len(self.preds)}"
        )

    def __len__(self):
        return len(self.input_texts)

    def _get_element_by_index(self, data_list, index):
        return data_list[index]

    def get_input_text(self, index):
        return self.input_texts[index]

    def get_label(self, index):
        return self._get_element_by_index(self.labels, index)

    def get_pred(self, index):
        return self._get_element_by_index(self.preds, index)

    def compute_mse(self, index):
        label = self.get_label(index)
        pred = self.get_pred(index)
        squared_errors = (label - pred) ** 2
        mse = np.mean(squared_errors)
        return mse

    def compute_sentence_mse_unreduced(self, index):
        label = self.get_label(index)
        pred = self.get_pred(index)
        squared_errors = (label - pred) ** 2
        return squared_errors

    def compute_mse_over_dataset(self):
        mse_list = []
        for i in range(len(self)):
            mse_list.append(self.compute_mse(i))
        return np.mean(mse_list)

    def compute_stats_per_word(self):
        word_mse = {}
        word_count = {}

        for i in range(len(self.input_texts)):
            words = self.input_texts[i].split(" ")
            labels = self.labels[i]
            preds = self.preds[i]

            for j, word in enumerate(words):
                mse = (labels[j] - preds[j]) ** 2
                if word in word_mse:
                    word_mse[word].append(mse)
                    word_count[word] += 1
                else:
                    word_mse[word] = [mse]
                    word_count[word] = 1

        mean_std_mse_per_word = {
            word: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": word_count[word],
            }
            for word, mse_list in word_mse.items()
        }
        return mean_std_mse_per_word

    def compute_stats_per_word_position(self):
        word_position_mse = {}
        word_position_count = {}

        for i in range(len(self.input_texts)):
            words = self.input_texts[i].split(" ")
            labels = self.labels[i]
            preds = self.preds[i]

            for j, (label, pred) in enumerate(zip(labels, preds)):
                mse = (label - pred) ** 2
                if j in word_position_mse:
                    word_position_mse[j].append(mse)
                    word_position_count[j] += 1
                else:
                    word_position_mse[j] = [mse]
                    word_position_count[j] = 1

        mean_std_mse_per_word_position = {
            position: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": word_position_count[position],
            }
            for position, mse_list in word_position_mse.items()
        }
        return mean_std_mse_per_word_position

    def compute_stats_per_sentence_length(self):
        sentence_length_mse = {}
        sentence_length_count = {}

        for i in range(len(self.input_texts)):
            sentence_length = len(self.input_texts[i].split(" "))
            mse = self.compute_mse(i)

            if sentence_length in sentence_length_mse:
                sentence_length_mse[sentence_length].append(mse)
                sentence_length_count[sentence_length] += 1
            else:
                sentence_length_mse[sentence_length] = [mse]
                sentence_length_count[sentence_length] = 1

        mean_std_mse_per_sentence_length = {
            length: {
                "mean": np.mean(mse_list),
                "std": np.std(mse_list),
                "count": sentence_length_count[length],
            }
            for length, mse_list in sentence_length_mse.items()
        }
        return mean_std_mse_per_sentence_length

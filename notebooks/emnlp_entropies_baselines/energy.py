#!/usr/bin/env python
# coding: utf-8

# ## Absolute Prominence Task - Differential Entropy and Control functions 

# In[2]:


from src.data.components.helsinki import HelsinkiProminenceExtractor
from src.data.components.datasets import TokenTaggingDataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import numpy as np
import os
from tqdm import tqdm

from src.utils.text_processing import python_lowercase_remove_punctuation
from src.utils.text_processing import get_wordlist_from_string

# only to create a valid dataset
dummy_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_special_tokens=True)


# ### Load data

# In[23]:


WAV_ROOT = "/nese/mit/group/evlab/u/luwo/projects/data/LibriTTS/LibriTTS"
LAB_ROOT = "/nese/mit/group/evlab/u/luwo/projects/data/LibriTTS/LibriTTSCorpusLabel/lab/word"
DATA_CACHE = "/nese/mit/group/evlab/u/luwo/projects/data/cache"

TRAIN_FILE = "train-clean-360"
VAL_FILE = "dev-clean"
TEST_FILE = "test-clean"

SAVE_DIR = "/nese/mit/group/evlab/u/luwo/projects/MIT_prosody/precomputed/emnlp/energy_mean"


# In[24]:


from src.data.energy_regression_datamodule import (
    EnergyRegressionDataModule as DataModule,
)


# In[32]:


dm = DataModule(
    wav_root=WAV_ROOT,
    lab_root=LAB_ROOT,
    data_cache=DATA_CACHE,
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
    test_file=TEST_FILE,
    dataset_name="libritts",
    model_name="gpt2",
    energy_mode="mean",
    score_last_token=True,
)


# In[33]:


dm.setup()


# In[34]:


train_texts, train_labels = dm.train_texts, dm.train_durations
val_texts, val_labels = dm.val_texts, dm.val_durations
test_texts, test_labels = dm.test_texts, dm.test_durations

print(
    f"Lengths of train, val, test in samples: {len(train_texts), len(val_texts), len(test_texts)}"
)


# In[35]:


GLOBAL_MEAN_PROMINENCE = np.mean([p for ps in train_labels for p in ps if p])
print("mean label", GLOBAL_MEAN_PROMINENCE)


# In[38]:


from src.utils.plots import plot_kde

labels_non_nan = [p for ps in train_labels for p in ps if p]

# plot_kde(
#     labels_non_nan,
#     label_name="Energy per Word",
#     title="Normalized Energy Distribution",
#     save_path="/Users/lukas/Desktop/projects/MIT/MIT_prosody/precomputed/predictions/energy_mean/energy_distribution.png",
# )


# In[39]:


train_texts[:3]


# In[40]:


train_labels[:3]


# In[42]:


from src.utils.text_processing import assign_labels_to_sentences

all_train_words, all_train_labels = assign_labels_to_sentences(
    train_texts, train_labels
)
all_test_words, all_test_labels = assign_labels_to_sentences(test_texts, test_labels)

print(f"Words and labels train: {len(all_train_words), len(all_train_labels)}")
print(f"Words and labels train: {len(all_test_words), len(all_test_labels)}")


# ### Kernel density estimation and Differential Entropy Computation

# In[43]:


# kernel density estimation
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

density = gaussian_kde(all_train_labels)

# xs = np.linspace(0, 6, 1000)
# plt.plot(xs, density(xs))
# plt.show()


# In[45]:


from src.utils.approximation import monte_carlo_diff_entropy

diff_entropy = monte_carlo_diff_entropy(density, all_train_labels, 50000)
print(f"Differential entropy: {diff_entropy:.4f}")


# # Baseline Models and Control Functions 

# ### Avg of all words in corpus

# In[46]:


avg_difference = np.mean(
    all_train_labels
)  # Here, train_labels are assumed to be prominences
print(f"Average prominence: {avg_difference}")

# compute mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

predictions = [avg_difference] * len(
    all_test_labels
)  # all_test_labels are assumed to be prominences
mse = mean_absolute_error(all_test_labels, predictions)
print(f"Mean absolute error: {mse}")

# compute r2
r2 = r2_score(all_test_labels, predictions)
print(f"R2 score: {r2}")

# compute pearson
pearson = pearsonr(all_test_labels, predictions)
print(f"Pearson correlation: {pearson}")

# store predictions
avg_test_predictions = []
for i in range(len(all_test_words)):
    sentence_predictions = [avg_difference] * len(all_test_words[i].split(" "))
    avg_test_predictions.append(sentence_predictions)

# store predictions
import pickle
import os

SAVE_DIR = "./path/to/save/directory"  # Please specify your directory path
os.makedirs(f"{SAVE_DIR}/avg", exist_ok=True)

with open(f"{SAVE_DIR}/avg/pred_avg.pkl", "wb") as f:
    pickle.dump(avg_test_predictions, f)

# store texts
with open(f"{SAVE_DIR}/avg/texts_avg.pkl", "wb") as f:
    pickle.dump(all_test_words, f)

# store labels
with open(f"{SAVE_DIR}/avg/labels_avg.pkl", "wb") as f:
    pickle.dump(all_test_labels, f)


# ### Corpus statistics: predict average per word 

# In[47]:


# collect the words types and their respective labels
word_prominence = {}
for word, prominence in zip(all_train_words, all_train_labels):
    if word not in word_prominence:
        word_prominence[word] = []
    word_prominence[word].append(prominence)

# compute the average prominence score for each word
word_prominence_avg = {}
for word, prominence in word_prominence.items():
    word_prominence_avg[word] = np.mean(prominence)

# for each word in the test set, get the average prominence score
predictions = []
for word in all_test_words:
    if word in word_prominence_avg:
        predictions.append(word_prominence_avg[word])
    else:
        predictions.append(avg_difference)  # avg_difference needs to be defined

print(f"Length of test set: {len(all_test_labels)}")
print(f"Length of predictions: {len(predictions)}")

# compute mae
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

mse = mean_absolute_error(all_test_labels, predictions)
print(f"Mean absolute error: {mse}")

# compute r2
r2 = r2_score(all_test_labels, predictions)
print(f"R2 score: {r2}")

# compute pearson
pearson = pearsonr(all_test_labels, predictions)
print(f"Pearson correlation: {pearson}")

# store predictions
word_test_predictions = []
for sentence in all_test_words:
    sentence_predictions = [
        word_prominence_avg[word] if word in word_prominence_avg else avg_difference
        for word in sentence.split()
    ]
    word_test_predictions.append(sentence_predictions)

# store predictions
import pickle
import os

SAVE_DIR = "./path/to/save/directory"  # Please specify your directory path
os.makedirs(f"{SAVE_DIR}/wordavg", exist_ok=True)

with open(f"{SAVE_DIR}/wordavg/pred_wordavg.pkl", "wb") as f:
    pickle.dump(word_test_predictions, f)

# store texts
with open(f"{SAVE_DIR}/wordavg/texts_wordavg.pkl", "wb") as f:
    pickle.dump(all_test_words, f)

# store labels
with open(f"{SAVE_DIR}/wordavg/labels_wordavg.pkl", "wb") as f:
    pickle.dump(all_test_labels, f)


# ## GloVe Baseline

# In[ ]:


GLOVE_PATH = "/Users/lukas/Desktop/projects/MIT/data/models/glove/glove.6B.300d.txt"

H_PARAMS = {
    "num_layers": 3,
    "input_size": 300,  # Update this based on the word embedding model
    "hidden_size": 128,
    "num_labels": 1,
    "dropout_probability": 0.1,
    "learning_rate": 0.001,
    "batch_size": 128,
    "max_epochs": 5,
}


# In[ ]:


from src.models.baselines.control_function import ControlFunction

control_function = ControlFunction(
    word_embedding_type="glove", word_embedding_path=GLOVE_PATH, hparams=H_PARAMS
)


# In[ ]:


control_function.fit(words=all_train_words, labels=all_train_labels)


# In[ ]:


# store predictions

pred = control_function.predict(all_test_words)
# flatten the pred
pred = [item for sublist in pred for item in sublist]


# In[ ]:


# compute mae
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(all_test_labels, pred)
print(f"Mean absolute error: {mae}")

# compute r2
from sklearn.metrics import r2_score

r2 = r2_score(all_test_labels, pred)
print(f"R2 score: {r2}")

# compute pearson
from scipy.stats import pearsonr

pearson = pearsonr(np.array(all_test_labels), pred)
print(f"Pearson correlation: {pearson}")


# In[ ]:





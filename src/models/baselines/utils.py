import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from GloVe import GloVeModel
from fastText import FastTextModel


def neg_log_likelihood(mu, sigma, label):
    dist = torch.distributions.Normal(mu, sigma)
    return -torch.sum(dist.log_prob(label))


def get_embedding_model(model_name: str, model_path: str):
    if model_name == "glove":
        return GloVeModel(model_path)
    elif model_name == "fasttext":
        return FastTextModel(model_path)
    else:
        raise ValueError(f"Model {model_name} not supported.")

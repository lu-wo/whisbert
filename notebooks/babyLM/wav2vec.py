#!/usr/bin/env python
# coding: utf-8

# In[121]:


from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices,
    _sample_negative_indices,
)

import torch
import numpy as np
from torch import nn


# In[122]:


bs, seqlen, feat = 16, 100, 78

preds = torch.rand(bs, seqlen, feat)
labels = torch.rand(bs, seqlen, feat)


# In[123]:


mask_time_indices = _compute_mask_indices(
    shape=(bs, seqlen), mask_prob=0.5, mask_length=1
)
mask_time_indices.shape


# In[124]:


sampled_negative_indices = _sample_negative_indices(
    features_shape=(bs, seqlen),
    num_negatives=20,
    mask_time_indices=mask_time_indices,
)
sampled_negative_indices = torch.from_numpy(sampled_negative_indices)
sampled_negative_indices.shape


# In[125]:


# for training, we sample negatives
# 3. sample K negatives (distractors) quantized states for contrastive loss

negative_features = labels.view(-1, feat)[sampled_negative_indices.long().view(-1)]
print(negative_features.shape)

negative_features = negative_features.view(bs, seqlen, -1, feat).permute(2, 0, 1, 3)
negative_features.shape


# In[126]:


def compute_contrastive_logits(
    target_features: torch.FloatTensor,
    negative_features: torch.FloatTensor,
    predicted_features: torch.FloatTensor,
    temperature: int = 0.1,
):
    """
    Compute logits for contrastive loss based using cosine similarity as the distance measure between
    `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
    """
    target_features = torch.cat([target_features, negative_features], dim=0)

    logits = torch.cosine_similarity(
        predicted_features.float(), target_features.float(), dim=-1
    ).type_as(target_features)

    # apply temperature
    logits = logits / temperature
    return logits


# In[127]:


# 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
# of equation (3) in https://arxiv.org/pdf/2006.11477.pdf

logits = compute_contrastive_logits(labels[None, :], negative_features, preds)
logits.shape


# In[128]:


# 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
# its cosine similarity will be masked

neg_is_pos = (preds == negative_features).all(-1)
neg_is_pos.shape


# In[129]:


if neg_is_pos.any():
    logits[1:, neg_is_pos] = float("-inf")


# In[130]:


logits.shape


# In[131]:


# 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
# -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
mask_time_indices = torch.from_numpy(mask_time_indices)

logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
logits.shape, target.shape


# In[132]:


contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
contrastive_loss


# In[ ]:





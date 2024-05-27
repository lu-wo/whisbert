#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


matched_path = "/om/user/luwo/projects/submissions/final_multi_23-29-25/csv/version_0/metrics.csv"
unmatched_path = "/om/user/luwo/projects/submissions/final_umatched_23-31-46/csv/version_0/metrics.csv"
text_path = "/om/user/luwo/projects/submissions/final_text_01-11-41/csv/version_0/metrics.csv"

# In[4]:

df_text = pd.read_csv(text_path)
df_matched = pd.read_csv(matched_path)
df_unmatched = pd.read_csv(unmatched_path)

print(f"text: {df_text.head()}")

print(f"matched", df_matched.head())


# In[5]:


print("unmatched", df_unmatched.head())


# In[6]:


def smooth_signal(x: np.array, window_len: int = 50):
    """smooth signal with a moving average filter"""
    return np.convolve(x, np.ones(window_len) / window_len, mode="same")


# In[7]:


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 1200

def plot_loss_functions(
    losses,
    names,
    y_logscale=False,
    x_logscale=False,
    y_label="Loss",
    x_label="Update steps",
    title="Model comparison",
    proc=None,
    save_path=None,
):
    assert len(losses) == len(names), "The number of losses and names should be equal"

    # losses = [loss[20:-20] for loss in losses]

    if proc is not None:
        losses = [proc(loss) for loss in losses]

    min_length = min(len(loss) for loss in losses)
    trimmed_losses = [loss[:min_length] for loss in losses]

    plt.figure(figsize=(10, 6))
    for loss, name in zip(trimmed_losses, names):
        plt.plot(loss[20:-20], label=name)

    if y_logscale:
        plt.yscale("log")
    if x_logscale:
        plt.xscale("log")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path)


# In[8]:


# plot mlm losses
plot_loss_functions(
    losses=[
        df_matched["train/loss_uni_mlm_step"],
        df_matched["train/loss_mm_mlm_step"],
        df_unmatched["train/loss_uni_mlm_step"],
        df_unmatched["train/loss_mm_mlm_step"],
    ],
    names=["unimodal", "multimodal", "unimodal unmatched", "multimodal unmatched"],
    proc=smooth_signal,
    save_path="multi_mlm_losses.png",
)


# In[10]:


# plot mam losses

plot_loss_functions(
    losses=[
        df_matched["train/loss_uni_mam_step"],
        df_matched["train/loss_mm_mam_step"],
        df_unmatched["train/loss_uni_mam_step"],
        df_unmatched["train/loss_mm_mam_step"],
    ],
    names=["unimodal", "multimodal", "unimodal unmatched", "multimodal unmatched"],
    proc=smooth_signal,
    save_path="multi_mam_losses.png",
)


plot_loss_functions(
    losses=[
        df_matched["train/loss_uni_mlm_step"],
        df_text["train/loss_uni_mlm_step"]
    ],
    proc=smooth_signal,
    names=["text-only", "multimodal"],
    save_path="text_vs_multimodal.png"
)

# In[ ]:





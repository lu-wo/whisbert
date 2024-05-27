#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.models.load_bert_whisper_multimodal_model import BertWhisperTrainingModule
import torch


# In[9]:


model_path = "/om/user/luwo/projects/MIT_prosody/logs/train/runs/2023-07-31/23-44-42/checkpoints/last.ckpt"


# In[10]:


model = BertWhisperTrainingModule.load_from_checkpoint(
    model_path, map_location=torch.device("cpu")
)


# In[11]:


bert = model.text_encoder.to("cpu")


# In[12]:


# store bert model under the same path

bert.save_pretrained(model_path.replace("last.ckpt", "bert"))


# In[13]:


whisper = model.audio_encoder.to("cpu")


# In[14]:


whisper.save_pretrained(model_path.replace("last.ckpt", "whisper"))


# ## Load Pre-Trained Whisper into WhisBert

# In[1]:


# model_path = "/Users/lukas/Desktop/Projects/MIT/MIT_prosody/precomputed/babylm_runs/23-00-09/checkpoints/whisper"


# In[7]:


# from src.models.bert_whisper_multimodal_model import BertWhisperTrainingModule
# from src.models.components.encoders import MyWhisperEncoder
# from transformers import WhisperConfig
# import torch

# cfg_path = "/Users/lukas/Desktop/Projects/MIT/MIT_prosody/precomputed/babylm_runs/23-00-09/checkpoints/whisper/config.json"
# weight_path = "/Users/lukas/Desktop/Projects/MIT/MIT_prosody/precomputed/babylm_runs/23-00-09/checkpoints/whisper/pytorch_model.bin"
# # load json as cfg dict
# cfg = WhisperConfig.from_pretrained(cfg_path)

# whisper = MyWhisperEncoder(cfg)
# whisper.load_state_dict(torch.load(weight_path))


# # In[9]:


# import os

# audio_encoder_path = "/Users/lukas/Desktop/Projects/MIT/MIT_prosody/precomputed/babylm_runs/23-00-09/checkpoints/whisper"

# print(f"Loading audio encoder from {audio_encoder_path}")
# audio_config = WhisperConfig.from_pretrained(
#     os.path.join(audio_encoder_path, "config.json")
# )
# audio_encoder = MyWhisperEncoder(audio_config).load_state_dict(
#     torch.load(os.path.join(audio_encoder_path, "pytorch_model.bin"))
# )


# # In[ ]:

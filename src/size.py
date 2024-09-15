#%%
import numpy as np
from transformers import ViTModel, ViTConfig

#%%
c = ViTConfig(image_size=(128,4352))
m = ViTModel(c)
f"{sum([np.prod(p.shape) for p in m.parameters()]):,.0f}"

#%%
c = ViTConfig(image_size=(128,500), num_hidden_layers=6)
m = ViTModel(c)
f"{sum([np.prod(p.shape) for p in m.parameters()]):,.0f}"

#%%
c
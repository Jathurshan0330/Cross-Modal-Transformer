from torch import Tensor
import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from torch.utils import data
import math
import random
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import glob
import os
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def softmax(data, dim=None, dtype=None):
    return torch.nn.functional.softmax(data, dim, dtype)

def scaled_dot_product_attention_mod(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    # attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,is_soft_max = True
) :
    """
    Reference : Pytorch
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)  ####Scaling part
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    # if attn_mask is not None:
    #     attn += attn_mask
    if is_soft_max:
        attn = softmax(attn, dim=-1)
    else:
        attn = attn/attn.max()
    # if dropout_p > 0.0:
    #     attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def plot_interpret(x,y,dydx,signal_type = "EEG", label = 0,save_path = None):


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # plt.figure(figsize = (30,5))
    plt.figure(figsize=(25,5))
    plt.plot(x,dydx)
    plt.title(f"Attention Map for Class {label}  {signal_type} ")
    plt.xlim(x.min(),x.max())
    # plt.colorbar()

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True,figsize = (30,10))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='Reds', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    fig.colorbar(line, ax=axs[0])
    fig.colorbar(line, ax=axs[1])
    axs[0].set_title(f"Class {label}  {signal_type} ")
    axs[1].set_title(f"Class {label}  {signal_type} ")

    axs[1].plot(x,y,linewidth = 2)
    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(y.min()-1,y.max()+1)
    # plt.show()
    if save_path:
        fig.savefig(os.path.join(save_path,f"{signal_type}_{label}"))
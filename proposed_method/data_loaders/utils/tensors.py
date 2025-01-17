import torch
from torch.utils import data
import os
import numpy as np
import random
from tqdm import tqdm
from loguru import logger

def t2m_collate(batch):
    batch = [{
        "text": b[0],
        "action": b[1],
        "random": b[2],
        "inp": b[3].T,
        "lengths": b[4],
    } for b in batch]

    return collate(batch)

def gt_collate(batch):
    batch = [{
        "text": b[0],
        "action": b[1],
        "random": b[2],
        "inp": b[3].permute(2, 1, 0),
        "lengths": b[4],
    } for b in batch]
    return collate(batch)

def collate(batch):
    not_none_batches = [b for b in batch if b is not None]

    data_batch = [b["inp"] for b in not_none_batches]

    if "lengths" in not_none_batches[0]:
        len_batch = [b["lengths"] for b in not_none_batches]
    else:
        len_batch = [len(b["inp"][0][0]) for b in not_none_batches]

    motions = collate_tensors(data_batch)
    lengths = torch.as_tensor(len_batch)
    temporal_masks = lengths_to_mask(lengths, motions.shape[-1]).unsqueeze(1)

    random_batch = [b["random"] for b in not_none_batches]
    random = collate_tensors(random_batch)
    

    if "text" in not_none_batches[0]:
        texts = [b["text"] for b in not_none_batches]

    if "action" in not_none_batches[0]:
        action_batch = [b["action"] for b in not_none_batches]
        actions = collate_tensors(action_batch)

    return texts, actions, motions, random, temporal_masks, lengths

def collate_tensors(batch):
    """
        Combines the batch list of different sequence length to one zero-padded tensor with the size of the longest sequence
    """
    # Get target tensor size
    dims = batch[0].dim()
    max_size= [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)

    # Create new zero filled tensor on same device and with same dtype as sequence elements with target size
    canvas = batch[0].new_zeros(size = size)

    # Fill new Tensor with list of sequence data
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
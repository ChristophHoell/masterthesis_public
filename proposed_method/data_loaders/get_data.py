"""
    Utility file,
    Creates the dataset object and wraps it into the dataloader object for batch wise access to the data
"""

import torch
import os
from loguru import logger

from torch.utils.data import DataLoader

from data_loaders.celebv_data import CelebVData, CelebVData_TextOnly 
from data_loaders.utils.tensors import collate, t2m_collate, gt_collate


def get_dataset_loader(args):
    opt_path = os.path.join(args.data_dir, "opt.txt")

    if args.split in ["train", "eval", "test", "all"]:
        split_file = args.split + ".txt"
    else:
        print(f"Split not possible")
        exit()

    if args.mode in ["train", "eval", "gt", "gt_normalized"]:
        dataset = CelebVData(opt_path, split_file, args.mode)
    elif args.mode == "text_only":
        dataset = CelebVData_TextOnly(opt_path, split_file)
    #elif args.mode == "uncond":
    #    dataset = Custom_Uncond_Dataset(opt_path, split_file, args.load_mode)
    else:
        logger.error(f"Dataset Mode currently not supported [{args.mode}], exiting")
        exit(0)

    if args.mode in ["gt", "gt_normalized"]:
        loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8, drop_last = True, collate_fn = gt_collate)
    else:
        loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8, drop_last = True, collate_fn = t2m_collate)
    return loader

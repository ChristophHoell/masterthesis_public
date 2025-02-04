"""
    Dataset Class Definition
"""

import torch
from torch.utils import data
import os
import numpy as np
import random as rand
from tqdm import tqdm
from loguru import logger
import pickle

from data_loaders.utils.get_opt import get_opt

# Used to provide an index for a action tag
# Required to enable filtering based on specific present actions
label_to_idx = {
    "talk": 0,
    "head_wagging": 1,
    "look_around": 2,
    "turn": 3,
    "shake_head": 4,
    "nod": 5,
    "blink": 6,
    "wink": 7,
    "squint": 8,
    "close_eyes": 9,
    "drink": 10,
    "sing": 11,
    "eat": 12,
    "smoke": 13,
    "listen_to_music": 14,
    "play_instrument": 15,
    "read": 16,
    "kiss": 17,
    "whisper": 18,
    "sneer": 19,
    "sigh": 20,
    "frown": 21,
    "weep": 22,
    "cry": 23,
    "smile": 24,
    "glare": 25,
    "gaze": 26,
    "laugh": 27,
    "shout": 28,
    "yawn": 29,
    "sneeze": 30,
    "cough": 31,
    "sleep": 32,
    "make_a_face": 33,
    "blow": 34,
    "sniff": 35,
    "chew": 36,
}

class CelebVData(data.Dataset):
    """
        Dataset Class Definition

        Required Parameters:
            opt_path:       Path to a opt.txt file defining the dataset parameters
            split_file:     Defines which dataset split should be used
            mode:           Defines the mode in which the data will be provided
    """
    def __init__(self, opt_path, split_file, mode):
        self.opt = get_opt(opt_path, "cuda:0")
        self.mode = mode
        self.load_mode = self.opt.load_mode


        with open(os.path.join(self.opt.data_root, split_file), "r") as f:
            names = f.read().splitlines()

        if len(names) > 100:
            assert self.load_mode == "disk", "The Dataset is too large to be loaded into RAM, please switch to loading from Disk"

        self.setup()
        self.prepare_data(names)



    def setup(self):
        """
            Performs the setup of the dataset
            - Loads the facial region indication (for potential loss weighting)
            - Loads the dataset mean and std
        """
        with open(os.path.join(self.opt.data_root, "vertex_id_to_face_region.pkl"), "rb") as f:
            region_ids = pickle.load(f)

        self.region_ids = {}
        for k in region_ids.keys():
            self.region_ids[k] = torch.tensor([[x*3, x*3 + 1, x*3 + 2] for x in region_ids[k]]).reshape(-1)
        
        self.mean = torch.load(os.path.join(self.opt.data_root, "mean.data"))
        self.std = torch.load(os.path.join(self.opt.data_root, "std.data"))

    def prepare_data(self, names):
        """
            Prepares the data by iterating once though the dataset metadata (not the motions directly)
            If desired, loads the entire dataset into RAM for faster access (and if the dataset is small enough)
            Otherwise the motions will be loaded from disk at runtime
        """
        data_dict = {}
        new_names = []
        video_ids = []

        for name in tqdm(names):
            try:
                metadata = torch.load(os.path.join(self.opt.data_root, "metadata", name + ".data"))

                # Update Length if not already present in metadata
                if metadata.get("length", None) is None:
                    metadata["length"] = metadata["boundaries"][1] - metadata["boundaries"][0]

                # Filter out too short sequences
                if metadata["length"] < self.opt.min_seq_len:
                    logger.error(f"Ignored sequence [{name}] for it is too short: {metadata['length']}")
                    continue

                # Filter out too long sequences
                if metadata["length"] > self.opt.max_seq_len:
                    logger.error(f"Ignored sequence [{name}] for it is too long: {metadata['length']}")
                    continue

                # Filter out too turned sequences
                num_too_turned_frames = sum(torch.logical_or(metadata["head_rotation"].T[1] < 145, metadata["head_rotation"].T[1] > 215))
                if num_too_turned_frames > self.opt.num_rotated_frames_threshold:
                    logger.error(f"Ignored sequence [{name}] for it is too turned - Number of too turned Frames: {num_too_turned_frames}")
                    continue

                text_data = []

                # Extract the text-description for the sample
                with open(os.path.join(self.opt.data_root, "actions", name + ".txt"), "r") as f:
                    for line in f.read().splitlines():
                        text_dict = {}
                        text_dict["caption"] = line.strip()
                        text_data.append(text_dict)

                # Error handling for non-existent descirption
                if len(text_data) == 0:
                    logger.error(f"Ignored sequence [{name}] for no corresponding annotation has been found")
                    continue

                # Filtering for "talking" description
                if self.opt.no_talking:
                    if any(["talking" in t["caption"] for t in text_data]):
                        logger.error(f"Skipped sequence [{name}] due to containing talking")
                        continue

                # Filtering for "cutted" sequences
                if self.opt.no_cuts:
                    if metadata["name"] in video_ids:
                        logger.error(f"Skipped sequence [{name}] due to being part of a cutted sequence")
                        continue
                    else:
                        video_ids.append(metadata["name"])

                # Extract per frame assigned action tags
                action = torch.zeros((metadata["length"], len(label_to_idx)))
                last = 0
                for k, v in metadata["timesteps"].items():
                    for e in v:
                        action[last:k, label_to_idx[e]] = 1
                    last = k

                # Filtering for "turning" action tags (turn, shake-head, ...)
                if self.opt.no_turned_desc:
                    turned_desc_ids = [1, 2, 3, 4, 5]
                    for i in turned_desc_ids:
                        if any(action[:, i]):
                            logger.error(f"Skipped Sequence due to containing [{i}]")
                            continue
                    
                # If desire, stores entire motion into RAM, else load at runtime
                if self.load_mode == "memory":
                    motion_data = torch.load(os.path.join(self.opt.data_root, "motions_vertices", name + ".motion"))
                    #motion = self.prepare_motion_data(motion_data)

                    data_dict[name] = {
                        "motion": motion_data,
                        "length": metadata["length"],
                        "text": text_data,
                        "action": action,
                        "random": torch.randn(self.opt.d_random),
                    }

                elif self.load_mode == "disk":
                    data_dict[name] = {
                        "length": metadata["length"],
                        "text": text_data, 
                        "action": action,
                        "random": torch.randn(self.opt.d_random),
                    }
                else:
                    logger.error(f"Dataset Loading mode is not defined [{self.load_mode}], exiting")
                    exit(0)

                new_names.append(name)
            except Exception as e:
                logger.error(f"Error in loading sequence: {e}")
                pass
                
        self.names_list = new_names
        self.data_dict = data_dict

        logger.info(f"Loaded Dataset, [{len(self.names_list)}] elements match requirements")

    def prepare_motion_data(self, motion_data):
        """
            Normalizes and reshapes the motion data to the correct dimensions (i.e. flatten the 5023x3 vertices to 15069)
        """
        motion = motion_data["vertex_offset"]
        motion = (motion - self.mean) / self.std
        
        motion = motion.reshape((motion_data["vertex_offset"].shape[0], -1))
        return motion

    def get_region_ids(self):
        return self.region_ids

    def normalize(self, data):
        return (data - self.mean) / self.std

    def inv_normalize(self, data):
        return data * self.std + self.mean

    def get_name(self, idx):
        return self.names_list[idx]

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        """
            Generator Callable, provides the data sample with index "idx"

            Extracts the requried data, then processes it according to the "mode" definition
            Zero-Pads the motion if it is too short 
        """
        name = self.names_list[idx]

        data = self.data_dict[name]

        m_length, text_list, action, random = data["length"], data["text"], data["action"], data["random"]

        if self.load_mode == "memory":
            motion_data = data["motion"]
        elif self.load_mode == "disk":
            motion_data = torch.load(os.path.join(self.opt.data_root, "motions_vertices", name + ".motion"))          

        text = rand.choice(text_list)
        caption = text["caption"]

        
        if self.mode in ["train", "eval"]:
            motion = self.prepare_motion_data(motion_data)
        elif self.mode == "gt":
            return caption, action, random, motion_data["vertex_offset"], m_length
        elif self.mode == "gt_normalized":
            motion = self.normalize(motion_data["vertex_offset"])
            return caption, action, random, motion, m_length

        # Zero Pad to max length
        if m_length < self.opt.max_seq_len:
            motion = torch.cat([motion, torch.zeros((self.opt.max_seq_len - m_length, motion.shape[1]))], axis = 0)
            action = torch.cat([action, torch.zeros((self.opt.max_seq_len - m_length, action.shape[1]))], axis = 0)

        return caption, action, random, motion, m_length

class CelebVData_TextOnly(data.Dataset):
    """
        Dataset Class Definition for a dataset without a gt motion assigned to it
        Necessary to have such a Dataset Object for the generation of new samples
    """
    def __init__(self, opt_path, split_file):
        self.opt = get_opt(opt_path, "cuda:0")

        with open(os.path.join(self.opt.data_root, split_file), "r") as f:
            names = f.read().splitlines()

        self.setup()
        self.prepare_data(names)



    def setup(self):
        """
            Loads the region ids
            Loads the dataset mean and std
        """
        with open(os.path.join(self.opt.data_root, "vertex_id_to_face_region.pkl"), "rb") as f:
            region_ids = pickle.load(f)

        self.region_ids = {}
        for k in region_ids.keys():
            self.region_ids[k] = torch.tensor([[x*3, x*3 + 1, x*3 + 2] for x in region_ids[k]]).reshape(-1)
        
        self.mean = torch.load(os.path.join(self.opt.data_root, "mean.data"))
        self.std = torch.load(os.path.join(self.opt.data_root, "std.data"))
        self.fixed_length = self.opt.max_seq_len

    def prepare_data(self, names):
        """
            Prepares the data by creating the dataset full of the description only entries
        """
        data_dict = {}
        new_names = []

        for name in tqdm(names):
            try:
                text_data = []

                with open(os.path.join(self.opt.data_root, "actions", name + ".txt"), "r") as f:
                    for line in f.read().splitlines():
                        text_dict = {}
                        text_dict["caption"] = line.strip()
                        text_data.append(text_dict)

                if len(text_data) == 0:
                    logger.error(f"Ignored sequence [{name}] for no corresponding annotation has been found")
                    continue

                new_names.append(name)
                data_dict[name] = {
                    "text": text_data,
                    "random": torch.randn(self.opt.d_random),
                }
            except Exception as e:
                logger.error(f"Error in loading sequence: {e}")
                pass
                
        self.names_list = new_names
        self.data_dict = data_dict

        logger.info(f"Loaded Dataset, [{len(self.names_list)}] elements match requirements")

    def get_region_ids(self):
        return self.region_ids

    def inv_normalize(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        """
            Generator Callable to obtain the description for the sample
        """
        name = self.names_list[idx]
        data = self.data_dict[name]

        text_list = data["text"]
        text = rand.choice(text_list)
        caption = text["caption"]
        random = data["random"]

        action = torch.zeros((self.fixed_length, len(label_to_idx)))
        motion = torch.zeros((self.fixed_length, self.opt.num_vertices * 3))

        return caption, action, random, motion, self.fixed_length


    

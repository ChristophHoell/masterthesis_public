import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm.auto import tqdm
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.motion_process import recover_from_ric
from datasets.utils.get_opt import get_opt
import os
import pickle
from loguru import logger
import random as rand

class Text2MotionDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    data_root = ''
    min_motion_len=40
    joints_num = None
    dim_pose = None
    max_motion_length = 196
    def __init__(self, opt, split, mode='train', accelerator=None):
        self.max_text_len = getattr(opt, 'max_text_len', 20)
        self.unit_length = getattr(opt, 'unit_length', 4)
        self.mode = mode
        motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        text_dir = pjoin(self.data_root, 'texts')

        if mode not in ['train', 'eval','gt_eval','xyz_gt','hml_gt']:
            raise ValueError(f"Mode '{mode}' is not supported. Please use one of: 'train', 'eval', 'gt_eval', 'xyz_gt','hml_gt'.")
        
        mean, std = None, None
        if mode == 'gt_eval':
            print(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
            # used by T2M models (including evaluators)
            mean = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            std = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['eval']:
            print(pjoin(opt.meta_dir, 'std.npy'))
            # used by our models during inference
            mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
            std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        else:
            # used by our models during train
            mean = np.load(pjoin(self.data_root, 'Mean.npy'))
            std = np.load(pjoin(self.data_root, 'Std.npy'))
            
        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate ours norms to theirs
            self.mean_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.eval_meta_dir, f'{opt.dataset_name}_std.npy'))
        if mode in ['gt_eval','eval']:
            self.w_vectorizer = WordVectorizer(opt.glove_dir, 'our_vab')
        
        data_dict = {}
        id_list = []
        split_file = pjoin(self.data_root, f'{split}.txt')
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list,disable=not accelerator.is_local_main_process if accelerator is not None else False):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < self.min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))
                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if mode=='train':
            if opt.dataset_name != 'amass':
                joints_num = self.joints_num
                # root_rot_velocity (B, seq_len, 1)
                std[0:1] = std[0:1] / opt.feat_bias
                # root_linear_velocity (B, seq_len, 2)
                std[1:3] = std[1:3] / opt.feat_bias
                # root_y (B, seq_len, 1)
                std[3:4] = std[3:4] / opt.feat_bias
                # ric_data (B, seq_len, (joint_num - 1)*3)
                std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
                # rot_data (B, seq_len, (joint_num - 1)*6)
                std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                            joints_num - 1) * 9] / 1.0
                # local_velocity (B, seq_len, joint_num*3)
                std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                        4 + (joints_num - 1) * 9: 4 + (
                                                                                                    joints_num - 1) * 9 + joints_num * 3] / 1.0
                # foot contact (B, seq_len, 4)
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

                assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            
            if accelerator is not None and accelerator.is_main_process:
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        "Z Normalization"
        if self.mode not in['xyz_gt','hml_gt']:
            motion = (motion - self.mean) / self.std

        "crop motion"
        if self.mode in ['eval','gt_eval']:
            # Crop the motions in to times of 4, and introduce small variations
            if self.unit_length < 10:
                coin2 = np.random.choice(['single', 'single', 'double'])
            else:
                coin2 = 'single'
            if coin2 == 'double':
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            elif coin2 == 'single':
                m_length = (m_length // self.unit_length) * self.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
        elif m_length >= self.max_motion_length:
            idx = random.randint(0, len(motion) - self.max_motion_length)
            motion = motion[idx: idx + self.max_motion_length]
            m_length = self.max_motion_length
        
        "pad motion"
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                        np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                        ], axis=0)
        assert len(motion) == self.max_motion_length


        if self.mode in ['gt_eval', 'eval']:
            "word embedding for text-to-motion evaluation"
            tokens = text_data['tokens']
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
        elif self.mode in ['xyz_gt']:
            "Convert motion hml representation to skeleton points xyz"
            # 1. Use kn to get the keypoints position (the padding position after kn is all zero)
            motion = torch.from_numpy(motion).float()
            pred_joints = recover_from_ric(motion, self.joints_num)  # (nframe, njoints, 3)  

            # 2. Put on Floor (Y axis)
            floor_height = pred_joints.min(dim=0)[0].min(dim=0)[0][1]
            pred_joints[:, :, 1] -= floor_height
            return pred_joints
    
        
        return caption, motion, m_length

class HumanML3D(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None):
        self.data_root = './data/HumanML3D'
        self.min_motion_len = 40
        self.joints_num = 22
        self.dim_pose = 263
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode HumanML3D %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode HumanML3D dataset ...' % mode)
        super(HumanML3D, self).__init__(opt, split, mode, accelerator)
        

class KIT(Text2MotionDataset):
    def __init__(self, opt, split="train", mode='train', accelerator=None):
        self.data_root = './data/KIT-ML'
        self.min_motion_len = 24
        self.joints_num = 21
        self.dim_pose = 251
        self.max_motion_length = 196
        if accelerator:
            accelerator.print('\n Loading %s mode KIT %s dataset ...' % (mode,split))
        else:
            print('\n Loading %s mode KIT dataset ...' % mode)
        super(KIT, self).__init__(opt, split, mode, accelerator)



            
class CelebVData(data.Dataset):
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
        with open(os.path.join(self.opt.data_root, "vertex_id_to_face_region.pkl"), "rb") as f:
            region_ids = pickle.load(f)

        self.region_ids = {}
        for k in region_ids.keys():
            self.region_ids[k] = torch.tensor([[x*3, x*3 + 1, x*3 + 2] for x in region_ids[k]]).reshape(-1)
        
        self.mean = torch.load(os.path.join(self.opt.data_root, "mean.data"))
        self.std = torch.load(os.path.join(self.opt.data_root, "std.data"))

    def prepare_data(self, names):
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

                with open(os.path.join(self.opt.data_root, "actions", name + ".txt"), "r") as f:
                    for line in f.read().splitlines():
                        text_dict = {}
                        text_dict["caption"] = line.strip()
                        text_data.append(text_dict)

                if len(text_data) == 0:
                    logger.error(f"Ignored sequence [{name}] for no corresponding annotation has been found")
                    continue

                if self.opt.no_talking:
                    if any(["talking" in t["caption"] for t in text_data]):
                        logger.error(f"Skipped sequence [{name}] due to containing talking")
                        continue

                if self.opt.no_cuts:
                    if metadata["name"] in video_ids:
                        logger.error(f"Skipped sequence [{name}] due to being part of a cutted sequence")
                        continue
                    else:
                        video_ids.append(metadata["name"])

                action = torch.zeros((metadata["length"], len(label_to_idx)))
                last = 0
                for k, v in metadata["timesteps"].items():
                    for e in v:
                        action[last:k, label_to_idx[e]] = 1
                    last = k

                if self.opt.no_turned_desc:
                    turned_desc_ids = [1, 2, 3, 4, 5]
                    for i in turned_desc_ids:
                        if any(action[:, i]):
                            logger.error(f"Skipped Sequence due to containing [{i}]")
                            continue
                    
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
            return caption, motion, m_length

        # Zero Pad to max length
        if m_length < self.opt.max_seq_len:
            motion = torch.cat([motion, torch.zeros((self.opt.max_seq_len - m_length, motion.shape[1]))], axis = 0)
            action = torch.cat([action, torch.zeros((self.opt.max_seq_len - m_length, action.shape[1]))], axis = 0)

        return caption, motion, m_length

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
import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np
import torch.nn as nn

import blobfile as bf
import torch
from torch.optim import AdamW
from tqdm import tqdm

from utils import logger, dist_util
from utils.model_util import load_model_wo_clip

from loguru import logger as log


class TrainLoop:
    def __init__(self, args, train_platform, model, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.batch_size = args.batch_size
        
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint

        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr_step_size = args.lr_step_size
        self.lr_step_reduction = args.lr_step_reduction

        self.step = 0
        self.resume_step = 0
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()
        self._load_and_sync_parameters()

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite
        
        self.opt = AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        if self.resume_step:
            self._load_optimizer_state()

        self.device = torch.device("cpu")
        if torch.cuda.is_available and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.l2_loss = nn.MSELoss(reduction = "none")
        self.emb_loss = nn.MSELoss()
        self.region_ids = data.dataset.get_region_ids()

    def _load_and_sync_parameters(self):
        """
            Loads a checkpoint if available
        """

        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.info(f"loading model from checkpoint: {resume_checkpoint}...")
            #self.model.load_state_dict(
            #    dist_util.load_state_dict(
            #        resume_checkpoint, map_location = dist_util.dev()
            #    )
            #)
            state_dict = torch.load(resume_checkpoint, map_location = "cpu")
            load_model_wo_clip(self.model, state_dict)
            self.model.to(dist_util.dev())

    def _load_optimizer_state(self):
        """
            Loads the optimizer state if available
        """

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        opt_checkpoint = os.path.join(os.path.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt")
        if os.path.exists(opt_checkpoint):
            logger.log(f"Loading Optimizer State from Checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location = dist_util.dev())
            self.opt.load_state_dict(state_dict)

    
    def run_loop(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch [{epoch}]")

            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                self.run_step(batch)

                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == "loss":
                            logger.info(f"Step [{self.step + self.resume_step}]: loss[{v:0.5f}]")
                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name = k, value = v, iteration = self.step, group_name = "Loss")
                
                if self.step % self.save_interval == 0:
                    self.save()

                self.step += 1

            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        """
            Performs a Forward - Backward pass
            then calls the optimization step
            then updates lr and logs the step
        """

        self.forward_backward(batch)
        self.opt.step()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch):
        (text, action, motion, random, mask, length) = batch

        self.opt.zero_grad()

        loss_dict = self.model(text, action, motion, random, mask)
        log_loss_dict(loss_dict)
        loss = loss_dict["full"]
        loss.backward()

    def _anneal_lr(self):
        """
            Updates the learning rate depending on the number of steps already trained (lr decay)
        """

        if not self.lr_anneal_steps and not self.lr_step_size:
            return

        elif (self.step + self.resume_step) == 0:
            return

        elif self.lr_anneal_steps:
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)

            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

        elif self.lr_step_size:
            if (self.step + self.resume_step) % self.lr_step_size == 0:
                for param_group in self.opt.param_groups:
                    param_group["lr"] = param_group["lr"] * self.lr_step_reduction

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"{(self.step + self.resume_step):09d}.pt"

    def save(self):
        state_dict = self.model.state_dict()
        clip_weights = [e for e in state_dict.keys() if e.startswith("clip_model.")]
        for e in clip_weights:
            del state_dict[e]

        logger.log(f"Saving model...")
        filename = self.ckpt_file_name()

        torch.save(state_dict, os.path.join(self.save_dir, f"model{filename}"))
        torch.save(self.opt.state_dict(), os.path.join(self.save_dir, f"opt{filename}"))


def parse_resume_step_from_filename(filename):
    """
        Parses filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the checkpoints number of steps
    """

    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try: 
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    """
        You can change this to be a separate path to save checkpoints to a blobstore or some external drive
    """
    return logger.get_dir()

def find_resume_checkpoint():
    """
        On your infrastructure, you may want to override this to automatically discover the latest checkpoint on your blobstorage, etc
    """
    return None

def log_loss_dict(losses):
    for k, v in losses.items():
        logger.logkv(k, v.mean().item())
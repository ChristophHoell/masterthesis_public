import os
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import json

from torch.optim import AdamW
from argparse import ArgumentParser, Namespace

from utils.fixseed import fixseed
from utils import dist_util
#from utils import logger as loss_logger
from data_loaders.get_data import get_dataset_loader
from train.train_platforms import TensorboardPlatform
#from render.render_head_mdm import Render_Head_MDM

class SimpleAutoencoder(nn.Module):
    def __init__(self, args):
        super(SimpleAutoencoder, self).__init__()
        self.device = args.device

        self.input_size = args.input_size
        self.latent_size = args.latent_size
        self.seq_len = args.seq_len

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.input_size),
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        bs = x.shape[0]
        x = x.permute(0, 3, 2, 1)
        x = x.flatten(start_dim = 2)
        x = x.to(self.device)
        loss = {}
        enc = self.encoder(x)
        rec = self.decoder(enc)

        loss["full"] = self.criterion(rec, x)

        return loss, rec

    def encode(self, x):
        bs = x.shape[0]
        #x = x.permute(0, 3, 2, 1)
        x = x.flatten(start_dim = 2)
        x = x.to(self.device)
        enc = self.encoder(x)
        return enc

class TrainLoop:
    def __init__(self, args, train_platform, model, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.data = data
        self.batch_size = args.batch_size

        self.lr = args.lr
        self.log_interval = args.log_interval
        
        self.weight_decay = args.weight_decay
        self.lr_step_size = args.lr_step_size
        self.lr_step_reduction = args.lr_step_reduction

        self.step = 0
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        self.device = torch.device("cpu")
        if torch.cuda.is_available and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.l2_loss = nn.MSELoss()

        self.renderer = load_renderer(args, args.save_dir)

    def run_loop(self):
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch [{epoch}]")

            for batch in tqdm(self.data):
                self.run_step(batch)

                if self.step % self.log_interval == 0:
                    for k,v in loss_logger.get_current().dumpkvs().items():
                        if k == "loss":
                            logger.info(f"Step [{self.step}]: loss[{v:0.5f}]")
                        if k in ["step", "samples"] or "_q" in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name = k, value = v, iteration = self.step, group_name = "Loss")
                        
                self.step += 1
        self.save()

        for k,v in loss_logger.get_current().dumpkvs().items():
            if k == "loss":
                logger.info(f"Step [{self.step}]: loss[{v:0.5f}]")
            if k in ["step", "samples"] or "_q" in k:
                continue
            else:
                self.train_platform.report_scalar(name = k, value = v, iteration = self.step, group_name = "Loss")

        i = 0
        for batch in self.data:
            if i == 10:
                break

            (texts, actions, motions, random, temporal_masks, lengths) = batch
            _, reconstructions = self.model(motions)
            reconstructions = reconstructions.reshape((len(texts), -1, self.data.dataset.opt.num_vertices, 3)).detach().cpu()
            motions = motions.permute((0, 3, 2, 1))

            for j, motion in enumerate(motions):
                if j > 10:
                    break
                name = texts[j].replace(" ", "_").replace(".", "")[:30] + f"_{j}"
                reconstruction = reconstructions[i]

                self.renderer.render_vertices(self.data.dataset.inv_normalize(motion), name + "_gt", texts[j])
                self.renderer.render_vertices(self.data.dataset.inv_normalize(reconstruction), name + "_rec", texts[j])


    def run_step(self, batch):
        self.forward_backward(batch)
        self.opt.step()
        self._anneal_lr()
        self.log_step()


    def forward_backward(self, batch):
        (_, _, motion, _, _, _) = batch

        self.opt.zero_grad()

        loss_dict, _ = self.model(motion)

        self.log_loss_dict(loss_dict)
        loss = loss_dict["full"]
        loss.backward()


    def _anneal_lr(self):
        if not self.lr_step_size:
            return

        elif self.step == 0:
            return

        elif self.lr_step_size:
            if (self.step % self.lr_step_size) == 0:
                for param_group in self.opt.param_groups:
                    param_group["lr"] = param_group["lr"] * self.lr_step_reduction

    def log_step(self):
        loss_logger.logkv("step", self.step)
        loss_logger.logkv("samples", (self.step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"{self.step:09d}.pt"

    def save(self):
        state_dict = self.model.state_dict()
        
        logger.info(f"Saving model...")

        filename = self.ckpt_file_name()

        torch.save(state_dict, os.path.join(self.save_dir, f"model{filename}"))

    @staticmethod
    def log_loss_dict(losses):
        for k, v in losses.items():
            loss_logger.logkv(k, v.mean().item())


def main(args):
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

    fixseed(args.seed)

    train_platform = TensorboardPlatform(args.save_dir)
    train_platform.report_args(args, name = "Args")

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError(f"save_dir [{args.save_dir}] already exists")
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent = 4, sort_keys = True)

    dist_util.setup_dist(args.device)

    logger.info("Creating Data Loader")

    data = get_dataset_loader(args)

    logger.info("Creating model")
    model = SimpleAutoencoder(args)

    model.to(dist_util.dev())

    logger.info(f"Total params {(sum(p.numel() for p in model.parameters()) / 1000000.0):.2f}M")
    logger.info(f"Training...")

    TrainLoop(args, train_platform, model, data).run_loop()
    train_platform.close()





def load_renderer(args, out_path):
    cfg = Namespace()
    
    cfg.image_size = 1024
    cfg.save_frames = False
    cfg.dist_factor = 1
    cfg.use_shape_template = False

    cfg.flame_geom_path = "./render/data/FLAME2020/generic_model.pkl"
    cfg.flame_lmk_path = "./render/data/landmark_embedding.npy"
    cfg.num_shape_params = 300
    cfg.num_exp_params = 100
    cfg.mesh_file = "./render/data/head_template_mesh.obj"
    cfg.video_format = "mp4v"
    cfg.fps = 30
    cfg.output_path = os.path.join(out_path, "render")

    os.makedirs(cfg.output_path, exist_ok = True)

    renderer = Render_Head_MDM(cfg)
    return renderer



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cuda", default = True, type = lambda x: x.lower() in ["true", "1"], help = "Defines if CUDA should be used")
    parser.add_argument("--device", default = 0, type = int, help = "ID of the CUDA device to be used")
    parser.add_argument("--seed", default = 10, type = int, help = "Randomness seed")
    parser.add_argument("--batch_size", default = 8, type = int, help = "Batch Size during the training")

    parser.add_argument("--input_size", default = 15069, type = int, help = "Size of the Input")
    parser.add_argument("--latent_size", default = 512, type = int, help = "Size of the Latent Space")

    parser.add_argument("--lr", default = 1e-4, type = float, help = "Learning Rate base")
    parser.add_argument("--lr_step_size", default = 2000, type = int, help = "Size of the Learning Rate steps")
    parser.add_argument("--lr_step_reduction", default = 0.5, type = float, help = "Learning Rate reduction factor at each full step")
    parser.add_argument("--weight_decay", default = 0.0, type = float, help = "Optimizer weight decay")
    parser.add_argument("--log_interval", default = 1000, type = int, help = "Interval at which the losses will be logged to console")
    parser.add_argument("--num_steps", default = 1000, type = int, help = "Number of training steps")


    parser.add_argument("--save_dir", required = True, type = str, help = "Location where the trained model should be stored")
    parser.add_argument("--overwrite", action = "store_true", help = "If true, will enable to use an existing save_dir")

    args = parser.parse_args()

    #args.data_dir = "./data/custom_dataset_reduced_90_frames/"
    args.data_dir = "./data/custom_single_short/"
    args.seq_len = 90
    args.mode = "gt_normalized"
    #args.split = "all"
    args.split = "train"


    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
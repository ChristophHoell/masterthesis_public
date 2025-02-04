"""
    Trains a model with the specified arguments

    Required Arguments:
        --save_dir


    Optional Arguments:
        --cuda
        --device
        --seed
        --batch_size

        --data_dir
        --split
        --mode
        --load_mode

        --num_layers
        --num_heads
        --latent_dim
        --activation
        --dropout
        --period
        --d_emb
        --num_embeddings
        --d_temporal
        --lambda_emb
        --teacher_forcing
        --ablation_use_style

        --overwrite
        --train_platform_type
        --lr
        --weight_decay
        --lr_anneal_steps
        --lr_step_size
        --lr_step_reduction
        --log_interval
        --save_interval
        --num_steps
        --resume_checkpoint

"""


import os
import json
import torch
from loguru import logger

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from utils.model_util import get_model_args
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

#from model.Model_new import Model
from model.final_model import Model
from train.training_loop import TrainLoop

def main():
    args = train_args()

    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

    fixseed(args.seed)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
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
    model_args = get_model_args(args, data.dataset)
    model_args.device = dist_util.dev()
    
    model = Model(model_args)

    model.to(dist_util.dev())

    logger.info(f"Total params {(sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0):.2f}M")
    logger.info(f"Training...")

    TrainLoop(args, train_platform, model, data).run_loop()
    train_platform.close()



if __name__ == "__main__":
    main()
    
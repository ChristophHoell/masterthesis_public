import os
import numpy as np
import torch
import torch.nn as nn
import shutil
from loguru import logger
from argparse import Namespace
from collections import OrderedDict
import json
import pickle

from datetime import datetime
from motion_loader import get_dataset_loader
from utils.metrics import *
from accelerate.utils import set_seed
from options.evaluate_options import TestOptions
from models.gaussian_diffusion import DiffusePipeline
from utils.model_load import load_model_weights
from models import build_models

def evaluate_rmse(name, gt_motion, sample_motion, f):
    #print("================= Evaluating MSE ==================")

    mse = nn.MSELoss()

    loss = torch.sqrt(mse(gt_motion, sample_motion))

    print(f"---> [{name}] - RMSE: {loss:.4f}")
    print(f"---> [{name}] - RMSE: {loss:.4f}", file = f, flush = True)

    return loss.item()

def evaluate_fid(name, gt_motion, sample_motion, f):
    eval_dict = OrderedDict({})

    #print("================= Evaluating FID ===============")

    gt_mu, gt_cov = calculate_activation_statistics(gt_motion)

    mu, cov = calculate_activation_statistics(sample_motion)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    print(f"---> [{name}] - FID: {fid:.4f}")
    print(f"---> [{name}] - FID: {fid:.4f}", file = f, flush = True)

    return fid.item()


def evaluate_diversity(name, motion, diversity_times, f):
    diversity = calculate_diversity(motion, diversity_times)

    print(f"---> [{name}] - Diversity: {diversity:.4f}")
    print(f"---> [{name}] - Diversity: {diversity:.4f}", file = f, flush = True)
    return diversity


def evaluate_l1_loss(name, gt_motion, sample_motion, f):
    l1 = nn.L1Loss()

    loss = l1(gt_motion, sample_motion)

    print(f"--> [{name}] - L1-Loss: {loss:.4f}")
    print(f"--> [{name}] - L1-Loss: {loss:.4f}", file = f, flush = True)

    return loss.item()


def evaluate_offset_to_base_face(name, base_face, sample_motion, f):
    l1 = nn.L1Loss()

    loss = l1(base_face, sample_motion)

    print(f"--> [{name}] - Base-Face-L1: {loss:.4f}")
    print(f"--> [{name}] - Base-Face-L1: {loss:.4f}", file = f, flush = True)

    return loss.item()

def evaluate(model, pipeline, dataloaders, args, region_ids, base_face):
    (gt_loader, eval_loader) = dataloaders
    base_face = select_vertices(gt_loader.dataset.normalize(base_face.unsqueeze(0).detach().cpu()), region_ids).repeat(len(gt_loader) * args.batch_size, 90, 1, 1).detach().cpu()


    all_metrics = OrderedDict({
        "RMSE": OrderedDict({}),
        #"FID": OrderedDict({}),
        "Diversity": OrderedDict({}),
        "L1": OrderedDict({}),
        "Base-Face-L1": OrderedDict({})
    })

    with open(args.log_file, "w") as f:

        for i in range(args.replication_times):
            print(f"====================== Replication Iteration [{i}] ============================")
            print(f"====================== Replication Iteration [{i}] ============================", file = f, flush = True)
            
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            rmse_loss_dict = OrderedDict({})
            fid_score_dict = OrderedDict({})
            diversity_dict = OrderedDict({})
            l1_loss_dict = OrderedDict({})
            l1_base_face = OrderedDict({})


            print(f"======================= Loading / Generating Samples ========================")
            print(f"======================= Loading / Generating Samples ========================", file = f, flush = True)
            
            if args.generate_samples:
                out_path = os.path.join(args.samples_path, f"rep_{i}.data")
                data = generate_samples(model, pipeline, gt_loader, out_path, args)
            else:
                data = torch.load(os.path.join(args.samples_path, f"rep_{i}.data"))

            data["ground_truth"] = select_vertices(data["ground_truth"], region_ids).detach().cpu()
            data["generated"] = select_vertices(data["generated"], region_ids).detach().cpu()

            print(f"Generated / Loaded Samples")
            print(f"Generated / Loaded Samples", file = f, flush = True)
            
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            print(f"======================= Evaluating  =========================================")
            print(f"======================= Evaluating  =========================================", file = f, flush = True)
            
            rmse_loss_dict["Model"] = evaluate_rmse("Model", data["ground_truth"], data["generated"], f)
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            l1_loss_dict["Model"] = evaluate_l1_loss("Model", data["ground_truth"], data["generated"], f)
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            """
            fid_score_dict["Model"] = evaluate_fid("Model", data["ground_truth"].numpy(), data["generated"].numpy(), f)
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)
            """

            diversity_dict["Dataset"] = evaluate_diversity("Dataset", data["ground_truth"].numpy(), args.diversity_times, f)
            diversity_dict["Model"] = evaluate_diversity("Model", data["generated"].numpy(), args.diversity_times, f)
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            l1_base_face["Dataset"] = evaluate_offset_to_base_face("Dataset", base_face, data["ground_truth"], f)
            l1_base_face["Model"] = evaluate_offset_to_base_face("Model", base_face, data["generated"], f)
            print(f"Time: {datetime.now()}")
            print(f"Time: {datetime.now()}", file = f, flush = True)

            
            print(f"======================= Evaluation Complete =================================")
            print(f"======================= Evaluation Complete =================================", file = f, flush = True)
            
            
            for key, item in rmse_loss_dict.items():
                if key not in all_metrics["RMSE"]:
                    all_metrics["RMSE"][key] = [item]
                else:
                    all_metrics["RMSE"][key] += [item]

            for key, item in l1_loss_dict.items():
                if key not in all_metrics["L1"]:
                    all_metrics["L1"][key] = [item]
                else:
                    all_metrics["L1"][key] += [item]
            
            """
            for key, item in fid_score_dict.items():
                if key not in all_metrics["FID"]:
                    all_metrics["FID"][key] = [item]
                else:
                    all_metrics["FID"][key] += [item]
            """
            for key, item in diversity_dict.items():
                if key not in all_metrics["Diversity"]:
                    all_metrics["Diversity"][key] = [item]
                else:
                    all_metrics["Diversity"][key] += [item]

            for key, item in l1_base_face.items():
                if key not in all_metrics["Base-Face-L1"]:
                    all_metrics["Base-Face-L1"][key] = [item]
                else:
                    all_metrics["Base-Face-L1"][key] += [item]


        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print(f"======================= {metric_name} Summary ===============================")
            print(f"======================= {metric_name} Summary ===============================", file = f, flush = True)

            for rep_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), args.replication_times)
                mean_dict[metric_name] = mean

                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f"---> [{rep_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}")
                    print(f"---> [{rep_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}", file = f, flush = True)

                elif isinstance(mean, np.ndarray):
                    line = f"---> [{rep_name}]"
                    for i in range(len(mean)):
                        line += f"(top {i+1}) Mean: {mean[i]:.4f} Cint: {conf_interval[i]:.4f}"
                    print(line)
                    print(line, file = f, flush = True)
    return mean_dict

def generate_samples(model, pipeline, dataloader, location, args):
    paths = []

    all_motions = []
    all_gt = []

    logger.info(f"Generating Samples...")


    for i, batch in enumerate(dataloader):
        (texts, gt_motions, m_lengths) = batch

        samples = pipeline.generate(texts, torch.LongTensor([int(x) for x in m_lengths]))[0]

        for p, s in enumerate(samples):
            motion = s.reshape((s.shape[0], 5023, 3))
            #motion = dataloader.dataset.inv_normalize(motion.detach().cpu())
            motion = motion.detach().cpu()
            all_motions.append(motion)

            all_gt.append(gt_motions[p])

    all_motions = torch.stack(all_motions)
    all_gt = torch.stack(all_gt)

    res = {
        "ground_truth": all_gt,
        "generated": all_motions,
    }

    torch.save(res, location)

    return res

def select_vertices(motions, region_ids):
    selected_regions = ["eye", "mouth"]

    selected_vertices = []
    for region in selected_regions:
        selected_vertices += region_ids[region]

    selected_vertices = torch.tensor(selected_vertices)

    return motions[:, :, selected_vertices]

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def load_dataset(args, mode):
    cfg = Namespace()
    cfg.dataset_name = "celebv"
    cfg.split = "test"
    cfg.mode = mode

    cfg.batch_size = args.batch_size
    cfg.data_dir = args.data_dir

    data = get_dataset_loader(cfg, batch_size = cfg.batch_size, split = cfg.split, mode = cfg.mode)
    return data


def load_args(path):
    model_args = Namespace()

    args_path = os.path.join(os.path.dirname(path), "args.json")
    assert os.path.exists(args_path), f"Arguments JSON not found! - {args_path}"

    with open(args_path, "r") as f:
        args_data = json.load(f)

    for k, v in args_data.items():
        setattr(model_args, k, v)

    return model_args


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    set_seed(0)

    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

    device_id = args.gpu_id
    device = torch.device("cuda:%d" % device_id if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    args.device = device

    args.cond_mask_prob = 0.0

    
    ckpt_path = os.path.join(args.model_dir, args.which_ckpt + ".tar")

    name = os.path.basename(os.path.dirname(args.model_dir))
    args.log_file = os.path.join(os.path.dirname(args.model_dir), f"eval_{name}_{args.which_ckpt}_{args.eval_mode}.log")
    args.samples_path = os.path.join(os.path.dirname(args.model_dir), f"eval_{name}_{args.which_ckpt}_samples")
    args.filter_region_pickle = os.path.join("./eval_model/vertex_id_to_face_region.pkl")
    os.makedirs(args.samples_path, exist_ok = True)

    if args.eval_mode == "debug":
        args.num_samples_limit = 200      # None means no limit (eval over all test split)
        args.diversity_times = 1
        args.replication_times = 1
        args.generate_samples = True

    elif args.eval_mode == "debug_disk":
        args.num_samples_limit = 200
        args.diversity_times = 100
        args.replication_times = 1
        args.generate_samples = False

    elif args.eval_mode == "full":
        args.num_samples_limit = 1000
        args.diversity_times = 150
        args.replication_times = 20
        args.generate_samples = True

    elif args.eval_mode == "full_disk":
        args.num_samples_limit = 1000
        args.diversity_times = 150
        args.replication_times = 20
        args.generate_samples = False

    else:
        logger.error(f"Eval mode does not match [{args.eval_mode}], exiting...")
        raise ValueError()

    logger.info("Creating DataLoader...")

    gt_loader = load_dataset(args, mode = "gt_normalized")
    eval_loader = load_dataset(args, mode = "eval")


    model = build_models(args)
    load_model_weights(model, ckpt_path, use_ema = not args.no_ema, device = device)

    pipeline = DiffusePipeline(
        opt = args,
        model = model,
        diffuser_name = args.diffuser_name,
        device = device,
        num_inference_steps = args.num_inference_steps,
        torch_dtype = torch.float32 if args.no_fp16 else torch.float16
    )

    with open(args.filter_region_pickle, "rb") as f:
        region_ids = pickle.load(f)
    base_face = torch.load(os.path.join(os.path.dirname(args.filter_region_pickle), "base_vertices.data"))

    evaluate(model, pipeline, (gt_loader, eval_loader), args, region_ids, base_face)

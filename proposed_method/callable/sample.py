# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
import torch
import shutil
from loguru import logger
from argparse import ArgumentParser
from types import SimpleNamespace

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import get_model_args, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.utils.tensors import collate
from render.render_head_mdm import Render_Head_MDM
#from model.head_mdm_new import Head_MDM
from model.final_model import Head_MDM

def main():
    args = generate_args()

    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)

    if args.use_fixseed:
        fixseed(args.seed)

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    data = load_dataset(args)
    max_frames = data.dataset.opt.max_seq_len
    fps = 30

    n_frames = min(max_frames, int(args.motion_length * fps))
    dist_util.setup_dist(args.device)

    if out_path == "":
        out_path = os.path.join(os.path.dirname(args.model_path), f"samples_{name}_{niter}_seed{args.seed}")

        if args.text_prompt != "":
            out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")
        elif args.input_text != "":
            out_path += "_" + os.path.basename(args.input_text).replace(".txt", "").replace(" ", "_").replace(".", "")

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    if args.text_prompt != "":
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != "":
        assert os.path.exists(args.input_text), f"Input Text File not found"
        with open(args.input_text, "r") as f:
            texts = f.readlines()
        texts = [s.replace("\n", "") for s in texts]
        args.num_samples = len(texts)

    if args.use_batched_sampling:
        texts_new = []
        tmp = []
        i = 0
        for text in texts:
            tmp.append(text)
            i += 1

            if i == args.batch_size:
                texts_new.append(tmp)
                i = 0
                tmp = []
        texts_new.append(tmp)
        texts = texts_new

    else:
        texts = [[t] for t in texts]

    if args.render_sample:
        renderer = load_renderer(args, out_path)
    total_num_samples = args.num_samples * args.num_repetitions

    model_args = get_model_args(args, data.dataset)
    model_args.device = dist_util.dev()
    model = Head_MDM(model_args)

    state_dict = torch.load(args.model_path, map_location = "cpu")
    load_model_wo_clip(model, state_dict)

    model.to(dist_util.dev())
    model.eval()

    logger.info(f"Setup Complete, Starting Sampling")

    all_motions = []
    all_texts = []

    for rep_i in range(args.num_repetitions):
        logger.info(f"Sampling repetitions [#{rep_i}]")

        for t in texts:
            args.batch_size = len(t)

            #x = torch.zeros((len(t), 15069, n_frames), device = dist_util.dev())
            with torch.no_grad():
                sample = model.predict(t, torch.randn(args.batch_size, data.dataset.opt.d_random))

            for i, s in enumerate(sample):
                motion = s.T.reshape((-1, data.dataset.opt.num_vertices, 3)).cpu()
                motion = data.dataset.inv_normalize(motion)
                text = t[i]

                all_motions.append(motion)
                all_texts.append(text)
                
                if args.render_sample:
                    name = text.replace(" ", "_").replace(".", "")[:30] + f"_{rep_i}"
                    renderer.render_vertices(motion, name, text)

            logger.info(f"Created {len(sample)} samples, total number: [{len(all_motions)}]")

    sequence_path = os.path.join(out_path, "results.frame")

    logger.info(f"Saving results to file [{sequence_path}]")

    data = {
        "motion": all_motions,
        "text": all_texts,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "data_format": "result_vertices",
    }

    torch.save(data, sequence_path)
    
    with open(sequence_path.replace(".frame", ".txt"), "w") as f:
        f.write("\n".join(all_texts))

def load_dataset(args):
    """
        Generates a Datset Object
    """

    args.split = "test"
    args.mode = "text_only"
    data = get_dataset_loader(args)
    return data

def load_renderer(args, out_path):
    cfg = SimpleNamespace()
    
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

if __name__ == "__main__":
    main()
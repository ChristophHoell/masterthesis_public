import os
import json
import torch
from loguru import logger
from argparse import ArgumentParser, Namespace

from render.render_Model import Render_Model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type = str, required = True, help = "Path to the results.frame file to be rendered")

    args = parser.parse_args()
   
    return args

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

    renderer = Render_Model(cfg)
    return renderer

def main(args):
    renderer = load_renderer(args, args.input_path)

    data = torch.load(os.path.join(args.input_path, "results.frame"))

    for i in range(len(data["motion"])):
        print(f"Processing sequence [{i}]...")
        vertex_offset = data["motion"][i]

        name = data["text"][i].replace(" ", "_").replace(".", "")[:30]
        renderer.render_vertices(data["motion"][i], name, data["text"][i])



if __name__ == "__main__":
    args = parse_args()
    main(args)
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger
from argparse import ArgumentParser

from flame.FLAME import FLAME
from tqdm import tqdm
import trimesh
import pyvista as pv
from pyvista.plotting.utilities import xvfb
xvfb.start_xvfb()
pv.start_xvfb()
import render.renderer as r
from copy import copy

class Renderer(object):
    def __init__(self, config, device = "cuda:0"):
        self.config = config
        self.device = device

        #self.faces = load_obj(self.config.mesh_file)
        #mesh = trimesh.load_mesh(self.config.mesh_file)
        #print(mesh)
        #self.faces = mesh.faces

        self.flame = FLAME(self.config).to(self.device)

        self.R = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]]).to(self.device)

        #self.diff_renderer = r.Renderer([self.config.image_size[0], self.config.image_size[1]], self.config.mesh_file).to(self.device)

        #self.faces = self.diff_renderer.faces[0].cpu().numpy()
        self.faces = np.load("./flame/data/faces_numpy.npy")

    def render_mesh_as_image(self, mesh):
        pl = pv.Plotter(off_screen = True, window_size = self.config.image_size)

        wrapped_mesh = pv.wrap(mesh.copy())

        pl.add_mesh(wrapped_mesh, color = "#a4dded", show_edges = self.config.show_wireframe, opacity = 1.0, smooth_shading = False)
        
        pl.set_background("white")

        if self.config.set_camera:
            pl.camera_position = (0, 0, 10)
            pl.camera.zoom(1.3)
            pl.camera.roll = 0
        
        pl.link_views()
        pl.render()

        image = np.asarray(pl.screenshot())
        pl.clear()
        return image

    def __call__(self, flame_parameters, name):
        if self.config.save_frames:
            os.makedirs(os.path.join(self.config.output_path, name), exist_ok = True)

        for k in flame_parameters.keys():
            flame_parameters[k] = flame_parameters[k].to(self.device)

        vertices, _, _ = self.flame(
            cameras = torch.inverse(self.R),
            shape_params = flame_parameters["shape"],
            expression_params = flame_parameters["exp"],
            eye_pose_params = flame_parameters["eyes"],
            jaw_pose_params = flame_parameters["jaw"],
            eyelid_params = flame_parameters["eyelids"],
        )

        images = []
        for i in tqdm(range(vertices.shape[0])):
            mesh = trimesh.Trimesh(faces = self.faces, vertices = vertices[i].cpu().numpy(), process = False)
            image = self.render_mesh_as_image(mesh) # 512, 512, 3

            frame_id = str(i).zfill(5)

            image = cv2.putText(image.copy(), name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), cv2.LINE_AA)

            images.append(image)

            if self.config.save_frames:
                cv2.imwrite(f"{self.config.output_path}/{name}/{frame_id}.jpg", image)

        out = cv2.VideoWriter(f"{self.config.output_path}/{name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.config.fps, self.config.image_size)
        for img in images:
            out.write(img)
        out.release()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Path to the Input file / folder")
    parser.add_argument("--output_path", default = "", type = str, help = "Path to the output folder (if none given, output will be returned in the input_path/render folder)")
    parser.add_argument("--image_size", default = 512, type = int, help = "Size of the Rendered Images")
    parser.add_argument("--save_frames", default = False, type = bool, help = "Flag if the Video Frames should be individually saved")
    parser.add_argument("--data_format", default = "mdm", type = str, help = "Additional information on the way the to be rendered data is stored")

    args = parser.parse_args()

    args.flame_geom_path = "./flame/data/FLAME2020/generic_model.pkl"
    args.flame_lmk_path = "./flame/data/landmark_embedding.npy"
    args.num_shape_params = 300
    args.num_exp_params = 100
    args.mesh_file = "./flame/data/head_template_mesh.obj"
    args.fps = 30

    if args.output_path == "":
        args.output_path = args.input_path + "render/"

    if os.path.isdir(args.input_path):
        args.input_path = os.path.join(args.input_path, "results.frame")
    
    args.image_size = (args.image_size, args.image_size)

    args.show_wireframe = False
    args.set_camera = True

    return args

def main():
    config = parse_args()
    renderer = Renderer(config)

    try:
        data = torch.load(config.input_path)
    except Exception as e:
        logger.error(f"Input Data could not be loaded! - Message: {e}")
        exit(0)

    os.makedirs(config.output_path, exist_ok = True)
    flame_params = []
    names = []
    prompts = []
    
    if config.data_format == "mdm":
        for i in range(len(data["motion"])):
            flame_params.append(data["motion"][i])

            name = data["text"][i].replace(" ", "_").replace(".", "") + "_" + str(i)

            if len(name) > 20:
                name_video = f"Sample_{i}"
            else:
                name_video = name

            
            names.append(name_video)
            prompts.append(f"{name_video} ----- {data['text'][i]}")

    elif config.data_format == "mica_processed":
        pass



    for i in range(len(flame_params)):
        renderer(flame_params[i], names[i])

    with open(os.path.join(config.output_path, "prompts.txt"), "w") as f:
        f.write("\n".join(prompts))


if __name__ == "__main__":
    main()
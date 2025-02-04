"""
    Renders the data from the .npy file to video sequences
    IMPORTANT: use MICA conda environment for rendering (not mdm) -> not yet compatible!!!
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from loguru import logger
from argparse import ArgumentParser

from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection

from flame.FLAME import FLAME
from render.renderer import Renderer

from tqdm import tqdm
import pickle


class Render_Model(object):
    """
        Config:
            input_path: path where the .npy file is

        Attributes to define:
            device, principal_point, focal_length, R, T, image_size

    """

    def __init__(self, config, device = "cuda:0"):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float)

        self.config = config
        self.device = device

        # Default / empty params
        self.image_size = torch.tensor([[config.image_size, config.image_size]]).cuda()
        self.focal_length = nn.Parameter(torch.tensor([[5000 / self.get_image_size()[0]]]).to(self.device))
        self.principal_point = nn.Parameter(torch.zeros(1, 2).float().to(self.device))


        dist = config.dist_factor * (0.75 * (1024 / config.image_size))
        R, T = look_at_view_transform(dist=dist)
        self.R = nn.Parameter(matrix_to_rotation_6d(R).to(self.device))
        self.T = nn.Parameter(T.to(self.device))

        self.cameras = PerspectiveCameras(
            device = self.device,
            principal_point = self.principal_point,
            focal_length = self.focal_length,
            R = rotation_6d_to_matrix(self.R), T = self.T,
            image_size = self.image_size
        )

        self.setup_renderer()

    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][0].item()

    def create_output_folders(self):
        Path(self.config.output_path).mkdir(parents = True, exist_ok = True)

    def setup_renderer(self):
        """
            Performs the renderer setup
            - Loads the 3D mesh object
            - defines the rasterization settings
            - places the lights such that the face is well lit
            - Creates the rasterizer and renderer
            - loads the vertices of the base-line face for later use
        """
        mesh_file = self.config.mesh_file

        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config).to(self.device)

        # Omitted: Flametext
        """
        self.flametex = FLAMETex(self.config).to(self.device)
        """    
        self.diff_renderer = Renderer(self.image_size, obj_filename = mesh_file).to(self.device)
        self.faces = load_obj(mesh_file)[1]

        raster_settings = RasterizationSettings(
            image_size = self.get_image_size(),
            faces_per_pixel = 1,
            cull_backfaces = True,
            perspective_correct = True
        )

        self.lights = PointLights(
            device = self.device,
            location = ((0.0, 0.0, 5.0), ),
            ambient_color = ((0.5, 0.5, 0.5), ),
            diffuse_color = ((0.5, 0.5, 0.5), )
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings = raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer = self.mesh_rasterizer,
            shader = SoftPhongShader(device = self.device, lights = self.lights)
        )

        vertices, _, _ = self.flame(
            cameras = torch.inverse(self.cameras.R),
            shape_params = torch.zeros((300, 1)).T.to(self.device),
            expression_params = torch.zeros((100, 1)).T.to(self.device),
            eye_pose_params = torch.zeros((12, 1)).T.to(self.device),
            jaw_pose_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unsqueeze(0).to(self.device),
            eyelid_params = torch.zeros((2, 1)).T.to(self.device),
        )

        self.base_vertices = vertices.type(torch.DoubleTensor).to(self.device)
        torch.save(self.base_vertices, "/mnt/e/base_vertices.data")
    

    def render_shape(self, vertices, faces = None, white = True):
        """
            Renders the provided vertices in white or light-blue
        """
        B = vertices.shape[0]
        V = vertices.shape[1]

        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)

        textures = TexturesVertex(verts_features = verts_rgb.cuda())
        meshes_word = Meshes(verts = [vertices[i] for i in range(B)], faces = [faces[i] for i in range(B)], textures = textures)

        blend = BlendParams(background_color = (1.0, 1.0, 1.0))

        fragments = self.mesh_rasterizer(meshes_word, cameras = self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_word, cameras = self.cameras, blend_params = blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()

        return rendering[:, 0:3, :, :]


    def __call__(self, flame_params, name):
        """
            Call for rendering of a motion sequence defined through flame-parameters
        """
        if self.config.save_frames:
            os.makedirs(os.path.join(self.config.output_path, name))

        shapes = flame_params["shape"].to(self.device)
        expressions = flame_params["exp"].to(self.device)
        eye_poses = flame_params["eyes"].to(self.device)
        jaw_poses = flame_params["jaw"].to(self.device)

        if "eyelids" in flame_params.keys():
            eyelid_params = flame_params["eyelids"].to(self.device)
        else:
            eyelid_params = None

        images = []

        # Iterate through each frame of the sequence
        for i in tqdm(range(len(shapes))):
            if eyelid_params == None:
                eye_param = None
            else:
                eye_param = eyelid_params[i].unsqueeze(0)


            self.diff_renderer.rasterizer.reset()
            self.diff_renderer.set_size(self.get_image_size())
            self.debug_renderer.rasterizer.raster_settings.image_size = self.get_image_size()

            vertices, lmk68, lmkMP = self.flame(
                cameras = torch.inverse(self.cameras.R),
                shape_params = shapes[i].unsqueeze(0),
                expression_params = expressions[i].unsqueeze(0),
                eye_pose_params = eye_poses[i].unsqueeze(0),
                jaw_pose_params = jaw_poses[i].unsqueeze(0),
                eyelid_params = eye_param
            )

            # Render the frame
            shape = self.render_shape(vertices, white = False)[0].cpu().numpy()
            frame_id = str(i).zfill(5)

            shape = (shape.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
            shape = np.minimum(np.maximum(shape, 0), 255).astype(np.uint8)

            # Add the description text to the top left of the video 
            shape = cv2.putText(shape.copy(), name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0, (255, 0, 0), 1, cv2.LINE_AA)

            images.append(shape)

            if self.config.save_frames:
                cv2.imwrite(f"{self.config.output_path}/{name}/{frame_id}.jpg", shape)

        # Store the rendered Video
        out = cv2.VideoWriter(f"{self.config.output_path}/{name}.avi", cv2.VideoWriter_fourcc(*self.config.video_format), self.config.fps, self.get_image_size())
        for i in range(len(images)):
            out.write(images[i])
        out.release()

    def render_like_mica(self, data):
        """
            Renders the frame but emulates head-movement through movement of the camera (as done in the MICA Facial Tracking project)
            Used to render the GT motions as MICA creates them for debug and demo purposes
        """
        camera = PerspectiveCameras(
            device = self.device,
            principal_point = torch.tensor(data["camera"]["pp"]),
            focal_length = torch.tensor(data["camera"]["fl"]),
            R = rotation_6d_to_matrix(torch.tensor(data["camera"]["R"])),
            T = torch.tensor(data["camera"]["t"]),
            image_size = self.image_size,
        ).to(self.device)

        vertices, _, _ = self.flame(
            cameras = torch.inverse(self.cameras.R),
            shape_params = torch.tensor(data["flame"]["shape"]).to(self.device),
            expression_params = torch.tensor(data["flame"]["exp"]).to(self.device),
            eye_pose_params = torch.tensor(data["flame"]["eyes"]).to(self.device),
            jaw_pose_params = torch.tensor(data["flame"]["jaw"]).to(self.device),
            eyelid_params = torch.tensor(data["flame"]["eyelids"]).to(self.device),
        )

        shape = self.render_shape(vertices, white = False)[0].cpu().numpy()

        shape = (shape.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
        shape = np.minimum(np.maximum(shape, 0), 255).astype(np.uint8)

        return shape

    def render_vertices(self, vertex_offset, name, text):
        """
            Renders the provided sequence of vertices defined as vertex-offset to the baseline-face
        """

        #R, _ = look_at_view_transform(dist = 2.0)       
        
        # defines the video-write depending on the desired output video-format
        if self.config.video_format == "mp4v":
            out = cv2.VideoWriter(f"{self.config.output_path}/{name}.mp4", cv2.VideoWriter_fourcc(*self.config.video_format), self.config.fps, self.get_image_size())
        elif self.config.video_format == "DVIX":
            out = cv2.VideoWriter(f"{self.config.output_path}/{name}.avi", cv2.VideoWriter_fourcc(*self.config.video_format), self.config.fps, self.get_image_size())
        else:
            logger.error(f"Video Format [{self.config.video_format}] not supported, exiting...")
            exit(0)

        # Iterate through the frames to render, add text and save the frames
        for i in range(len(vertex_offset)):
            offset = vertex_offset[i].unsqueeze(0).to(self.device)
            vertices = (self.base_vertices + offset)
            vertices = vertices.type(torch.FloatTensor).to(self.device)

            self.diff_renderer.rasterizer.reset()
            self.diff_renderer.set_size(self.get_image_size())
            self.debug_renderer.rasterizer.raster_settings.image_size = self.get_image_size()
            shape = self.render_shape(vertices, white = False)[0].cpu().numpy()
            shape = (shape.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
            shape = np.minimum(np.maximum(shape, 0), 255).astype(np.uint8)
            shape = cv2.putText(shape.copy(), text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            #print(f"Added Text: {name}")
            #images.append(shape)
            out.write(shape)
        out.release()
        
        
        #for i in range(len(images)):
        #    out.write(images[i])
        #out.release()






def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Path to the ###.frame file containing the motion sequences to be rendered")
    parser.add_argument("--output_path", default = "", type = str, help = "Path where the rendered motion sequences are to be stored")
    parser.add_argument("--image_size", default = 1024, type = int, help = "Size of the to be generated videos")
    parser.add_argument("--save_frames", default = False, type = bool, help = "Determines if the video frames should be saved individually")
    parser.add_argument("--data_format", default = "mdm", type = str, help = "Format in which the flame parameters are stored, possible ['mdm', 'mica_processed', 'mica_raw']")
    parser.add_argument("--combine_videos", default = False, type = bool, help = "Combine the rendered videos into one large video?")
    parser.add_argument("--dist_factor", default = 1, type = int, help = "Multiplicative Factor on the distance between camera and face")
    parser.add_argument("--use_shape_template", default = False, type = bool, help = "Render with the Shape Template")

    args = parser.parse_args()

    args.flame_geom_path = "./data/FLAME2020/generic_model.pkl"
    args.flame_lmk_path = "./data/landmark_embedding.npy"
    args.num_shape_params = 300
    args.num_exp_params = 100
    args.mesh_file = "./data/head_template_mesh.obj"
    args.video_format = "DIVX"
    args.fps = 30

    if args.output_path == "":
        args.output_path = args.input_path + "render/"

    # Default settings not to be changed with command line args


    return args



def main():
    config = parse_args()
    renderer = Render_Model(config)

    # Define slight differences in the rendering process depending on how the data was generated (Diffusion (MDM), MICA), MICA (turning), ...)
    if config.data_format == 'mdm':
        data = torch.load(os.path.join(config.input_path, "results.frame"))

        os.makedirs(config.output_path, exist_ok = True)
        names = []
        prompts = []


        for i in range(len(data["motion"])):
            print(f"Processing sequence [{i}]...")   
            flame_params = data["motion"][i]

            name = data["text"][i].replace(" ", "_").replace(".", "") + "_" + str(i)

            if len(name) > 30:
                name_video = f"Sample_{i}"
            else:
                name_video = name


            renderer(flame_params, name_video)
            names.append(name_video)
            prompts.append(f"{name_video} ---- {data['text'][i]}")


        with open(os.path.join(config.output_path, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts))
        
        if config.combine_videos:
            for i, name in enumerate(names):
                if i == 0:
                    in_1 = name + ".avi"
                    continue
                in_2 = name + ".avi"
                out = f"tmp_{i}.mp4"

                if i == len(names) - 1:
                    out = "final.mp4"

                print(f"Combining {in_1} and {in_2}")
                os.system(f"sudo ffmpeg -i {config.output_path}{in_1} -i {config.output_path}{in_2} -filter_complex hstack -c:v libx264 -preset slow -crf 5 -c:a aac -movflags +faststart {config.output_path}{out} -y")
                if i > 1:
                    os.system(f"rm {config.output_path}{in_1}")
                in_1 = out
                
    elif config.data_format == "mica_processed":
        sequences = os.listdir(config.input_path)

        for i, sequence in tqdm(enumerate(sequences)):
            try:
                data = torch.load(os.path.join(config.input_path, sequence))
                if config.data_usage == "no_jaw":
                    data["jaw"] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unsqueeze(0).repeat(300, 1)
                    data["shape"] = torch.zeros((300,300)).T
                elif config.data_usage == "no_shape":
                    data["shape"] = torch.zeros((300,300)).T
                elif config.data_usage == "jaw_only":
                    data["exp"] = torch.zeros((100, 300)).T
                    data["shape"] = torch.zeros((300, 300)).T
                    data["eyes"] = torch.zeros((12, 300)).T
                    data["eyelids"] = torch.zeros((2, 300)).T

                renderer(data, sequence.replace(".frame", ""))   
            except:
                pass

    elif config.data_format == "mica_raw":
        frames = os.listdir(os.path.join(config.input_path, "checkpoint"))

        out = cv2.VideoWriter(f"{os.path.join(config.output_path, 'render.avi')}", cv2.VideoWriter_fourcc(*config.video_format), config.fps, (512, 512))

        for i, frame in tqdm(enumerate(frames)):
            data = torch.load(os.path.join(config.input_path, "checkpoint", frame))
            out.write(renderer.render_like_mica(data))

        out.release()

    elif config.data_format == "ground_truth_variations":
        sequences = os.listdir(os.path.join(config.input_path, "motions"))
        os.makedirs(config.output_path, exist_ok = True)

        for i, sequence in enumerate(sequences):
            try:
                data = torch.load(os.path.join(config.input_path, "motions", sequence))

                if "text" in data.keys():
                    text = data["text"][0]
                else:
                    with open(os.path.join(config.input_path, "actions", sequence.replace(".seq", ".txt"))) as f:
                        text = f.read()
                
                if len(text) < 10:
                    name = text
                else:
                    name = f"GT_{str(i).zfill(4)}"

                if "motion" in data.keys():
                    motion = data["motion"][0]
                else:
                    motion = {
                        "exp": data["exp"],
                        "shape": data["shape"],
                        "tex": data["tex"],
                        "eyes": data["eyes"],
                        "eyelids": data["eyelids"],
                        "jaw": data["jaw"],
                    }

                renderer(motion, name)
            except Exception as e:
                print(e)
                continue

    elif config.data_format == "vertices":
        data = torch.load(os.path.join(config.input_path))
        vertex_offset = data["vertex_offset"]

        os.makedirs(config.output_path, exist_ok = True)

        if config.use_shape_template:
            vertex_offset += data["shape_offset"]

        renderer.render_vertices(vertex_offset, config.input_path.split("/")[-1].replace(".motion", ""))

    elif config.data_format == "result_vertices":
        data = torch.load(os.path.join(config.input_path, "results.frame"))

        os.makedirs(config.output_path, exist_ok = True)
        names = []
        prompts = []

        for i in range(len(data["motion"])):
            print(f"Processing sequence [{i}]...")  
            vertex_offset = data["motion"][i]

            name = data["text"][i].replace(" ", "_").replace(".", "") + "_" + str(i)

            if len(name) > 20:
                name_video = f"Sample_{i}"
            else:
                name_video = name

            renderer.render_vertices(vertex_offset, name_video)

            names.append(name_video)
            prompts.append(f"{name_video} ---- {data['text'][i]}")


        with open(os.path.join(config.output_path, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts))
        
        
        if config.combine_videos:
            for i, name in enumerate(names):
                if i == 0:
                    in_1 = name + ".avi"
                    continue
                in_2 = name + ".avi"
                out = f"tmp_{i}.mp4"

                if i == len(names) - 1:
                    out = "final.mp4"

                print(f"Combining {in_1} and {in_2}")
                os.system(f"sudo ffmpeg -i {config.output_path}{in_1} -i {config.output_path}{in_2} -filter_complex hstack -c:v libx264 -preset slow -crf 5 -c:a aac -movflags +faststart {config.output_path}{out} -y")
                if i > 1:
                    os.system(f"rm {config.output_path}{in_1}")
                in_1 = out

    elif config.data_format == "voca_test":
        vertex_offset = torch.load(os.path.join(config.input_path))
        renderer.render_vertices(vertex_offset, config.input_path.split("/")[-1].replace(".motion", ""))

    else:
        logger.error(f"Data Format not supported [{config.data_format}]")

if __name__ == "__main__":
    main()
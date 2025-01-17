import os
from loguru import logger
import torch
from argparse import ArgumentParser
import numpy as np
import trimesh
from pytorch3d.io import load_obj


def main(args):
    data = torch.load(args.input_path)
    base_vertice = torch.load(args.base_vertice_location)

    verts, faces, aux = load_obj(args.template_location)
    faces = faces.verts_idx.cpu().numpy()

    desired_meshes = [0, 10, 20, 30, 40, 50, 60, 70, 80]

    for i in range(len(data["motion"])):
        print(f"Processing sequence [{i}]...")

        vertex_offset = data["motion"][i]

        vertice = base_vertice + vertex_offset

        name = data["text"][i].replace(" ", "_").replace(".", "").replace(",", "")[:30]

        os.makedirs(os.path.join(args.output_path, f"{name}_{i}"), exist_ok = True)

        #for desired in desired_meshes:
        #    mesh = vertice[desired].numpy()
        #    out_path = os.path.join(args.output_path, name, f"{str(desired).zfill(4)}.ply")
        #    trimesh.Trimesh(faces = faces, vertices = mesh, process = False).export(out_path, encoding="ascii")

        for j, mesh in enumerate(vertice):
            mesh = mesh.numpy()
            out_path = os.path.join(args.output_path, f"{name}_{i}", f"{str(j).zfill(4)}.ply")
            trimesh.Trimesh(faces = faces, vertices = mesh, process = False).export(out_path, encoding = "ascii")











def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Path of the results.frame file")
    parser.add_argument("--output_path", default = None, type = str, help = "Path to the output location. Defaults to the input location")

    args = parser.parse_args()

    args.template_location = os.path.join("/mnt", "e", "TMP", "results", "head_template_mesh.obj")
    args.base_vertice_location = os.path.join("/mnt", "e", "TMP", "results", "base_vertices.data")

    if "results.frame" in args.input_path:
        args.output_path = os.path.dirname(args.input_path)
    else:
        args.output_path = args.input_path
        args.input_path = os.path.join(args.input_path, "results.frame")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
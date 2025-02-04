"""
    Script File
    Takes specified frames (10, 20, ...) of multiple videos and combines them into one large image
    Used for figure generation in the thesis (not used in the end)

"""
import os
import cv2
from loguru import logger
from pathlib import Path
from glob import glob
from argparse import ArgumentParser
import numpy as np

ffmpeg_location = "/usr/bin/ffmpeg"

def split_video(input_path, output_path):
    if os.path.exists(output_path):
        return sorted(glob(f"{output_path}/*.png"))
    os.makedirs(output_path)

    os.system(f"{ffmpeg_location} -i '{input_path}' -vf fps=30 -q:v 1 '{output_path}/%05d.png' -loglevel quiet")

    images = sorted(glob(f"{output_path}/*.png"))
    for img in images:
        i = cv2.imread(img)
        i[0:100, ...] = [250, 250, 250]
        cv2.imwrite(img, i)

    return images

def merge_images(images):
        
    grid = np.concatenate(images, axis = 1)
    return grid


def to_image(img):
    img = (img.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    return img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Path to the folder containing the input videos")
    parser.add_argument("--output_path", default = None, type = str, help = "Path to the folder where the output files should be stored. If set to none, will be the same as input path.")
    
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.input_path

    return args


def main(args):
    video_names = os.listdir(args.input_path)

    frame_nums = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    img_size = 1024


    for name in video_names:
        current = os.path.join(args.input_path, name)
        target_path = os.path.join(args.output_path, name.replace(".mp4", ""))

        frames = split_video(current, target_path)
        logger.critical(f"Frames num: {len(frames)}")
        selected = []
        for i in frame_nums:
            selected.append(frames[i])

        images = []

        for f in selected:
            img = cv2.imread(f)
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

            images.append(img)

        combined = merge_images(images)
        cv2.imwrite(f"{args.input_path}/{name.replace('.mp4', '')}.png", combined)






if __name__ == "__main__":
    args = parse_args()
    main(args)

















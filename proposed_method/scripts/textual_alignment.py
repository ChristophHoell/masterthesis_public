"""
    Adds the assigned action class as a subscript to each frame of the input video
    Used in the presentation for demo purposes
"""

import os
from loguru import logger
from glob import glob
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import cv2


def main(args):
    images = sorted(glob(f'{args.input_path}/*.png'))

    for i, img in enumerate(tqdm(images)):
        img = cv2.imread(img)
        h,w,c = img.shape
        new = np.ones((h + 100, w, c), img.dtype) * 255
        new[:h,:w,:c] = img

        text = []
        for anno in args.annotation:
            if (args.fps * anno[1]) <= i and (args.fps * anno[2]) >= i:
                text.append(anno[0])


        text = " ".join(text)
        logger.info(f"Frame: {i} - Anno: {text}")
        new = cv2.putText(new.copy(), text, (20, 570), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imwrite(f"{args.output_path}/{i}.png", new)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", required = True, type = str, help = "Input Path")


    args = parser.parse_args()
    args.fps = 30
    args.annotation = [['talk', 0, 11], ['gaze', 0, 11], ['frown', 2, 3], ['blink', 4, 11]]
    args.output_path = os.path.join(args.input_path, "result")
    os.makedirs(args.output_path, exist_ok = True)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
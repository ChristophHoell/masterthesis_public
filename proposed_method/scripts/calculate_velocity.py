import os
import torch
from loguru import logger
from argparse import ArgumentParser, Namespace

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type = str, required = True, help = "Path to the result.frame file")

    args = parser.parse_args()
    return args

def main(args):
    data = torch.load(args.input_path)

    vels = []

    for m in data["motion"]:
        vel = m[1:] - m[:-1]
        vels.append(vel.mean())
        logger.critical(f"Average Velocity: {vel.mean()}")

    vels = torch.tensor(vels)
    logger.critical(f"Total Average Velocity: {vels.mean()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
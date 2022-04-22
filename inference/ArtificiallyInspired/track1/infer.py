import argparse
import os
import glob

import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm

from dataset import MultiResolutionDataset
from inference.utils import G


def infer(args, dataset, g):
    os.makedirs(args.output, exist_ok=True)
    for item_idx in tqdm(range(len(dataset))):
        g.infer_and_save(dataset, item_idx)


def run_inference_on_input(args, g):
    imgs = glob.glob(os.path.join(args.input, "*.png"))
    imgs = sorted([i for i in imgs if ("_segm" not in i) and ("_mask" not in i)])
    assert len(imgs) > 0
    df = pd.DataFrame(imgs, columns=["Image"])
    df["Mask"] = df["Image"].apply(lambda x: x.replace(".png", "_mask.png"))
    df["Output"] = df["Image"].apply(lambda x: os.path.join(args.output, os.path.basename(x)))

    for i in range(len(df)):
        assert os.path.exists(df.loc[i, "Image"])    
        assert os.path.exists(df.loc[i, "Mask"])    

    dataset = MultiResolutionDataset(df, resolution=args.size)

    infer(args, dataset, g)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument('-i', '--input_path', type=str, default='', required=True, help='test data root path')
    parser.add_argument('-d', '--dataset', type=str, default='', required=True, help='dataset name')
    parser.add_argument('-o', '--output', type=str, default='./output_ntire/', help='output path')
    parser.add_argument("--size", type=int, default=512, help="image sizes for the model")
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g = G(args, device)
    dataset_path = os.path.join(args.input_path, args.dataset)
    out_root = args.output
    for s in glob.glob(os.path.join(dataset_path, "*/*")):
        args.input = s
        args.output = os.path.join(out_root, args.dataset, '/'.join(args.input.split('/')[-2:]))
        print(f"Inferring for {args.input} and saving to {args.output}")
        run_inference_on_input(args, g)

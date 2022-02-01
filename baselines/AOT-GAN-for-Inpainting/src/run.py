import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args
from utils.painter import Sketcher

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


@torch.no_grad()
def demo(args):
    # load images 
    img_list = []
    for ext in ['*.jpg', '*.png']: 
        img_list.extend(glob(os.path.join(args.dir_image, ext)))
    
    png_paths = sorted([i for i in img_list if 'mask' not in i])
    mask_paths = sorted([i for i in img_list if 'mask' in i])    

    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train, map_location='cpu'))
    model.to(device)
    model.eval()
    os.makedirs(args.dir_output, exist_ok=True)
    for i, m in tqdm(zip(png_paths, mask_paths), total=len(png_paths)):
        filename = os.path.basename(i)
        orig_img = cv2.resize(cv2.imread(i, cv2.IMREAD_COLOR), (args.image_size, args.image_size))
        img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0).to(device)
        h, w, c = orig_img.shape
        mask = cv2.resize(cv2.imread(m, cv2.IMREAD_GRAYSCALE), (args.image_size, args.image_size))

        mask_tensor = (ToTensor()(mask)).unsqueeze(0).to(device)
        # import ipdb; ipdb.set_trace()
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

        pred_np = postprocess(pred_tensor[0])
        out = os.path.join(args.dir_output, filename)                                                                                                                                                          
        cv2.imwrite(out, pred_np)          


if __name__ == '__main__':
    demo(args)

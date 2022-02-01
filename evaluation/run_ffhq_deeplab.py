# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# This code is a modification of the https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/run_deeplab.py
# which is a modification of the main.py file
# from the https://github.com/chenxi116/DeepLabv3.pytorch repository

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import evaluation.ffhq_deeplab as deeplab
from utils import download_file
import requests

CLASSES = ['background' ,'skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']
deeplab_file_spec = dict(file_url='https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY', 
        file_path='evaluation/ffhq_deeplab_model/deeplab_model.pth', 
        file_size=464446305, 
        file_md5='8e8345b1b9d95e02780f9bed76cc0293'
)

class FFHQDeeplab(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = deeplab.resnet101(
    	      pretrained=False,
    	      num_classes=len(CLASSES),
    	      num_groups=32,
    	      weight_std=True,
    	      beta=False
        )
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        model_fname = deeplab_file_spec['file_path']
        if not os.path.isfile(model_fname):
            print('Downloading FFHQ DeeplabV3 Model parameters')
            with requests.Session() as session:
                download_file(session, deeplab_file_spec)

            print('Done!')
        checkpoint = torch.load(model_fname)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        self.model.load_state_dict(state_dict)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, filename):        
        input_image = Image.open(filename)
        _size = input_image.size
        input_image = input_image.convert("RGB")
        input_image=input_image.resize((513,513),Image.BILINEAR)
        input_tensor = self.preprocess(input_image)
        input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        input_tensor = input_tensor.to(self.device)
        outputs = self.model(input_tensor)
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        mask_pred = Image.fromarray(pred)
        mask_pred = np.array(mask_pred.resize((_size), Image.NEAREST))
        return mask_pred

    def show_result(self, filename_gt, segm, out_file=None, opacity=0.3):
        input_image = Image.open(filename_gt)
        img = np.array(input_image.convert("RGB"))
        img = img.copy()

        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(len(CLASSES), 3))
        np.random.set_state(state)
        palette = np.array(palette)

        color_seg = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[segm == label, :] = color
    
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        if out_file is not None:
            Image.fromarray(img).save(out_file)
        else:
            return img

def main():
    import argparse
    import os
    import requests
    from data_loader import CelebASegmentation
    resnet_file_spec = dict(file_url='https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM', file_path='evaluation/deeplab_model/R-101-GN-WS.pth.tar', file_size=178260167, file_md5='aa48cc3d3ba3b7ac357c1489b169eb32')

    resolution = args.resolution
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model_fname = 'deeplab_model/deeplab_model.pth'
    dataset_root = 'ffhq_aging{}x{}'.format(resolution,resolution)
    assert os.path.isdir(dataset_root)
    dataset = CelebASegmentation(dataset_root, crop_size=513)

    if not os.path.isfile(resnet_file_spec['file_path']):
        print('Downloading backbone Resnet Model parameters')
        with requests.Session() as session:
            download_file(session, resnet_file_spec)

        print('Done!')

    model = getattr(deeplab, 'resnet101')(
    	      pretrained=True,
    	      num_classes=len(dataset.CLASSES),
    	      num_groups=32,
    	      weight_std=True,
    	      beta=False)

    model = model.cuda()
    model.eval()
    if not os.path.isfile(deeplab_file_spec['file_path']):
        print('Downloading DeeplabV3 Model parameters')
        with requests.Session() as session:
            download_file(session, deeplab_file_spec)

        print('Done!')

    checkpoint = torch.load(model_fname)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    model.load_state_dict(state_dict)
    for i in range(len(dataset)):
        inputs=dataset[i]
        inputs = inputs.cuda()
        outputs = model(inputs.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
        pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        imname = os.path.basename(dataset.images[i])
        mask_pred = Image.fromarray(pred)
        mask_pred=mask_pred.resize((resolution,resolution), Image.NEAREST)
        try:
            mask_pred.save(dataset.images[i].replace(imname,'parsings/'+imname[:-4]+'.png'))
        except FileNotFoundError:
            os.makedirs(os.path.join(os.path.dirname(dataset.images[i]),'parsings'))
            mask_pred.save(dataset.images[i].replace(imname,'parsings/'+imname[:-4]+'.png'))

        print('processed {0}/{1} images'.format(i + 1, len(dataset)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=256,
                        help='segmentation output size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    args = parser.parse_args()
    main()

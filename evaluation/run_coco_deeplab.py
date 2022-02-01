# This code uses the MMSegmentation framework to deploy DeepLabV3 trained on CocoStuff-164k
# https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3

import torch
import torch.nn as nn
from mmseg.apis import inference_segmentor, init_segmentor
from PIL import Image
import numpy as np

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
        "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "branch", "bridge", "building-other", 
        "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", 
        "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", 
        "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", 
        "furniture-other", "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat", "metal", 
        "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement", "pillow", "plant-other", 
        "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", 
        "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs", "stone", "straw", 
        "structural-other", "table", "tent", "textile-other", "towel", "tree", "vegetable", "wall-brick", 
        "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", 
        "waterdrops", "window-blind", "window-other", "wood"]

class CocoDeeplab(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        config_file = 'evaluation/mmsegmentation/configs/deeplabv3/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k.py'
        checkpoint_file = 'evaluation/mmsegmentation/checkpoints/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth'

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(config_file, checkpoint_file, device=device)
    
    @torch.no_grad()
    def forward(self, filename):
        result = inference_segmentor(self.model, filename)
        return result[0].astype(np.uint8)

    def show_result(self, filename_gt, segm, out_file=None, opacity=0.3):
        return self.model.show_result(filename_gt, [segm], out_file=out_file, opacity=opacity)
        
if __name__ == '__main__':
    # test a single image and show the results
    img = 'src.jpg'  # or img = mmcv.imread(img), which will only load it once
    print(Image.open(img).size)
    model = CocoDeeplab()
    result = model(img)
    # visualize the results in a new window
    # model.show_result(img, result, show=False)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    model.show_result(img, result, out_file='segmentation.png', opacity=0.3)
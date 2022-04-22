from io import BytesIO
import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


def resize_real_mask(real, mask, segm):
    assert real.shape[0] == mask.shape[0], f"{real.shape}, {mask.shape}"
    assert real.shape[1] == mask.shape[1], f"{real.shape}, {mask.shape}"
    assert real.shape[0] == segm.shape[0], f"{real.shape}, {segm.shape}"
    assert real.shape[1] == segm.shape[1], f"{real.shape}, {segm.shape}"
    height, width = real.shape[0], real.shape[1]
    if min(height, width) >= 512: 
        return real, mask, segm, [0, height, 0, width] 
    top_pad = max(512 - height, 0) // 2 
    bottom_pad = max(512 - top_pad - height, 0)
    left_pad = max(512 - width, 0) // 2 
    right_pad = max(512 - left_pad - width, 0)
    real = cv2.copyMakeBorder(real, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    mask = cv2.copyMakeBorder(mask, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    segm = cv2.copyMakeBorder(segm, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT)

    return real, mask, segm, [top_pad, top_pad + height, left_pad, left_pad + width]


def get_cropped_real_mask(real, mask, segm, bbox):
    top, bottom, left, right = bbox
    real_box = real[top:bottom, left:right]
    mask_box = mask[top:bottom, left:right]
    segm_box = segm[top:bottom, left:right]
    return real_box, mask_box, segm_box


def get_top_left_lines(dim, resolution):
    assert dim >= resolution
    lines = []
    c = (dim - resolution) // 2
    lines.append(c)
    while c > 0:
        c -= resolution // 2
        lines.append(c if c > 0 else 0)
    c = (dim - resolution) // 2
    while c + resolution < dim:
        c += resolution // 2
        lines.append(c if c + resolution < dim else dim - resolution)
    return lines


def weave_lines(tls, lls):
    boxes = []
    for tl in tls:
        for ll in lls:
            boxes.append((tl, ll))
    return sort_boxes(boxes, 0, (tls[0], lls[0]))

def sort_boxes(boxes, idx, center):
    if len(boxes) <= 1:
        return boxes
    distinct = list(np.unique(np.array([b[idx] for b in boxes])))
    distinct = sorted(distinct)
    opp_idx = 1 if idx == 0 else 0
    if len(distinct) == 1:
        return sort_boxes(boxes, opp_idx, center)
    
    mid_idx = len(distinct) // 2
    if distinct[mid_idx] > center[idx]:
        mid_idx -= 1
    mid = distinct[mid_idx]
    prev_boxes, mid_boxes, after_boxes = [], [], []
    for b in boxes:
        if b[idx] < mid:
            prev_boxes.append(b)
        elif b[idx] > mid:
            after_boxes.append(b)
        else:
            mid_boxes.append(b)    
    sorted_boxes = sort_boxes(mid_boxes, idx, center) + sort_boxes(prev_boxes, idx, center) + sort_boxes(after_boxes, opp_idx, center)
    return sorted_boxes


def get_coordinates(imshape, resolution):
    height, width = imshape[0], imshape[1]
    top_lines = get_top_left_lines(height, resolution)
    left_lines = get_top_left_lines(width, resolution)
    boxes = weave_lines(top_lines, left_lines)
    boxes = [(b[0], b[0] + resolution, b[1], b[1] + resolution) for b in boxes]
    return boxes


def fill_mask(mask, all_boxes, box_idx):
    for i in range(box_idx):
        b = all_boxes[i]
        mask[b[0]:b[1], b[2]:b[3]] = 0
    return mask

class MultiResolutionDataset(Dataset):
    def __init__(self, data_df, transform=None, segm_classes=19, resolution=512):
        self.data_df = data_df
        self.transform = transform
        print("Number of images: ", len(self.data_df))
        self.segm_classes = segm_classes

        for i in range(len(self.data_df)):
            img_path = self.data_df.loc[i, "Image"]
            assert os.path.exists(img_path)
            m_path = self.data_df.loc[i, "Mask"]
            if not pd.isna(m_path): assert os.path.exists(m_path)
            s_path = self.data_df.loc[i, "Segm"]
            if not pd.isna(s_path): assert os.path.exists(s_path)

        self.resolution = resolution

    def __len__(self):
        return len(self.data_df)
    
    def onehot_initialization_v2(self, a):
        out = np.zeros( (a.size,self.segm_classes), dtype=np.uint8)
        out[np.arange(a.size),a.ravel()] = 1
        out.shape = a.shape + (self.segm_classes,)
        return out
    
    def load_image(self, img_path, mask=False):
        img = cv2.imread(img_path)
        if mask:
            img = img[:,:,0]
        return img
    
    def get_item(self, idx, box_idx, is_every_nn):
        return self.__getitem__(idx, box_idx, is_every_nn)

    def __getitem__(self, idx, box_idx=0, is_every_nn=False):
        if box_idx == 0 or is_every_nn:
            img_path = self.data_df.loc[idx, "Image"]
        else:
            img_path = self.data_df.loc[idx, "Output"]
        img = self.load_image(img_path)
        img = img[:,:,::-1].copy()

        segm_path = self.data_df.loc[idx, "Segm"]
        segm = self.load_image(segm_path, mask=True)
        
        mask_path = self.data_df.loc[idx, "Mask"]
        mask = self.load_image(mask_path, mask=True)
        
        img, mask, segm, orig_box = resize_real_mask(img, mask, segm) # Resize in aspect ratio with the smaller side as 512
        all_boxes = get_coordinates(img.shape, self.resolution)
        if not is_every_nn:
            mask = fill_mask(mask, all_boxes, box_idx)
        crop_img, mask, segm = get_cropped_real_mask(img, mask, segm, all_boxes[box_idx])

        assert len(segm.shape) == 2
        one_hot_segm = self.onehot_initialization_v2(segm)
        one_hot_segm = np.moveaxis(one_hot_segm, 2, 0)
        assert one_hot_segm.shape == (self.segm_classes, self.resolution, self.resolution), one_hot_segm.shape

        if self.transform is not None:
            img = self.transform(img)
            crop_img = self.transform(crop_img)
        assert crop_img.shape == (3, self.resolution, self.resolution), crop_img.shape
        
        mask = mask < 127  # mask == 0 should be removed from image
        mask = mask[None]
        assert mask.shape == (1, self.resolution, self.resolution), mask.shape

        return img, crop_img, mask.astype('float32'), one_hot_segm.astype('float32'), segm.astype('float32'), orig_box, all_boxes, idx

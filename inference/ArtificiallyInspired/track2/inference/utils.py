import os
import numpy as np
import cv2
import PIL.Image
import torch
from torchvision import utils

from comodgan import Generator
from unetsepconv import unet_sep_conv

def get_is_every_n(same, diff, same_sum, diff_sum):
    if len(same) != 1:
        return False
    if len(diff) != 2 or (diff == 0).sum() == 0:
        return False
    if abs(diff_sum[:10].mean() - 0.5) == 0.5 or abs(diff_sum[-10:].mean() - 0.5) == 0.5:
        return False
    return True

def get_is_nn(same, diff, same_sum, diff_sum):
    if len(same) != 2 or ((same == 0).sum() == 0 and (same == 0).sum() == 1):
        return False
    if len(diff) != 2 or ((diff == 0).sum() == 0 and (diff == 0).sum() == 1):
        return False

    mid = len(diff_sum) // 2
    if mid < 5:
        return False
    if abs(diff_sum[:10].mean() - 0.5) == 0.5 or abs(diff_sum[-10:].mean() - 0.5) == 0.5 or abs(diff_sum[mid-5:mid+5].mean() - 0.5) == 0.5:
        return False
    return True

def is_mask_every_nn(impath):
    img = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    img = img / 255.
    same_sum = img.mean(0)
    diff_sum = img.mean(1)
    same = np.unique(same_sum)    
    diff = np.unique(diff_sum)
    if len(same) > len(diff):
        same_sum, diff_sum = diff_sum, same_sum
        same, diff = diff, same
    
    is_evaery_n = get_is_every_n(same, diff, same_sum, diff_sum)
    is_nn = get_is_nn(same, diff, same_sum, diff_sum)
    return is_evaery_n or is_nn

class G:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        g_ema = Generator(
            latent_size=512, mapping_layers=8, segm_classes=args.segm_classes
        ).to(device)
        ckpt = torch.load(f"./checkpoint/track2_{args.dataset}_comodgan.pt", map_location=lambda storage, loc: storage)
        g_ema.load_state_dict(ckpt["g_ema"])
        g_ema.eval()
        g_unet = unet_sep_conv().to(device)
        ckptu = torch.load(f"./checkpoint/track2_{args.dataset}_unet.pth", map_location=lambda storage, loc: storage)
        g_unet.load_state_dict(ckptu["model"])
        g_unet.eval()

        self.g_ema = g_ema
        self.g_unet = g_unet
    
    def infer_and_save(self, dataset, item_idx):
        with torch.no_grad():
            sample_z = torch.randn(1, self.args.latent, device=self.device)
            box_idx = 0
            is_every_nn = is_mask_every_nn(dataset.data_df.loc[item_idx, "Mask"])
            all_preds = []
            while box_idx == 0 or box_idx < len(all_boxes):
                full_img, real_img, mask, oh_segm, segm, orig_box, all_boxes, item_idx = dataset.get_item(item_idx, box_idx, is_every_nn)
                full_img = torch.Tensor(full_img[None]).to(self.device)
                real_img = torch.Tensor(real_img[None]).to(self.device)
                mask = torch.Tensor(mask[None]).to(self.device)
                oh_segm = torch.Tensor(oh_segm[None]).to(self.device)
                segm = torch.Tensor(segm[None]).to(self.device)

                if is_every_nn:
                    sample = self.g_unet(real_img, mask)
                    assert len(sample.shape) == 4, sample.shape
                    assert sample.shape[0] == 1, sample.shape
                    all_preds.append(sample)
                else:
                    sample, _, _ = self.g_ema(real_img, oh_segm, mask, [sample_z])
                    assert len(sample.shape) == 4, sample.shape
                    assert sample.shape[0] == 1, sample.shape
                    b = all_boxes[box_idx]
                    full_img[:,:,b[0]:b[1], b[2]:b[3]] = sample
                    full_img = full_img[:,:,orig_box[0]:orig_box[1],orig_box[2]:orig_box[3]]
                    save_path = dataset.data_df.loc[item_idx, "Output"]
                    utils.save_image(full_img[0], save_path, nrow=1, normalize=True, range=(-1, 1))
                box_idx += 1
            
            if is_every_nn:
                full_img = torch.zeros_like(full_img)
                for b, sample in zip(all_boxes[::-1], all_preds[::-1]):
                    full_img[:,:,b[0]:b[1], b[2]:b[3]] = sample
                full_img = full_img[:,:,orig_box[0]:orig_box[1],orig_box[2]:orig_box[3]]
                save_path = dataset.data_df.loc[item_idx, "Output"]
                utils.save_image(full_img[0], save_path, nrow=1, normalize=True, range=(-1, 1))

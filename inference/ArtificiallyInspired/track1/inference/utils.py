import os
import numpy as np
import cv2
import PIL.Image


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
        self.device = device
        import torch
        from unetsepconv import unet_sep_conv
        g_unet = unet_sep_conv().to(device)
        ckptu = torch.load(f"./checkpoint/track1_{args.dataset}_unet.pth", map_location=lambda storage, loc: storage)
        g_unet.load_state_dict(ckptu["model"])
        g_unet.eval()
        Gs = self.load_tf_model(f"./checkpoint/track1_{args.dataset}_comodgan.pkl")
        
        self.g_unet = g_unet
        self.Gs = Gs
    
    def load_tf_model(self, checkpoint): 
        from dnnlib import tflib
        from training import misc
        tflib.init_tf()
        _, _, Gs = misc.load_pkl(checkpoint)
        return Gs
    
    def infer_and_save(self, dataset, item_idx):
        box_idx = 0
        is_every_nn = is_mask_every_nn(dataset.data_df.loc[item_idx, "Mask"])
        all_preds = []
        while box_idx == 0 or box_idx < len(all_boxes):
            full_img, real_img, mask, orig_box, all_boxes, item_idx = dataset.get_item(item_idx, box_idx, is_every_nn)
            if is_every_nn:
                import torch
                with torch.no_grad():
                    real_img = torch.Tensor(real_img[None]).to(self.device)
                    mask = torch.Tensor(mask[None]).to(self.device)

                    sample = self.g_unet(real_img, mask)
                    assert len(sample.shape) == 4, sample.shape
                    assert sample.shape[0] == 1, sample.shape
                    all_preds.append(sample)
            else:
                from training import misc
                latent = np.random.randn(1, *self.Gs.input_shape[1:])
                sample = self.Gs.run(latent, None, real_img[None], 
                                        mask[None], 
                                        truncation_psi=1)
                assert len(sample.shape) == 4, sample.shape
                assert sample.shape[0] == 1, sample.shape

                b = all_boxes[box_idx]
                full_img[:,b[0]:b[1], b[2]:b[3]] = sample[0]
                full_img = full_img[:,orig_box[0]:orig_box[1],orig_box[2]:orig_box[3]]
                save_path = dataset.data_df.loc[item_idx, "Output"]
                full_img = misc.adjust_dynamic_range(full_img, [-1, 1], [0, 255])
                full_img = full_img.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
                full_img = PIL.Image.fromarray(full_img)
                full_img.save(save_path)
            box_idx += 1
        
        if is_every_nn:
            from torchvision import utils
            full_img = torch.zeros((1, *full_img.shape), dtype=torch.float32)
            for b, sample in zip(all_boxes[::-1], all_preds[::-1]):
                full_img[:,:,b[0]:b[1], b[2]:b[3]] = sample
            full_img = full_img[:,:,orig_box[0]:orig_box[1],orig_box[2]:orig_box[3]]
            save_path = dataset.data_df.loc[item_idx, "Output"]
            utils.save_image(full_img[0], save_path, nrow=1, normalize=True, range=(-1, 1))

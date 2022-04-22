import numpy as np
import sys
import os
sys.path.append('.')
from mask_generation.utils import MaskGeneration, MergeMask
from PIL import Image
from shutil import copy
import random
# random.seed(1)
# np.random.seed(1)

def main(dataset, mask_mode, number, save=True):
    folder = 'test'
    filename = os.path.join('datasets', dataset, folder+'_gt', mask_mode, str(number).zfill(6)+'.png')
    # out = np.concatenate((gt, mask), axis=1)
    # mask -> 255: inpaint, 0: keep
    mask_filename = os.path.join('datasets', dataset, folder, mask_mode, str(number).zfill(6)+'_mask.png')
    array = np.array(Image.open(filename).convert('RGB'))
    mask = np.array(Image.open(mask_filename).convert('RGB'))
    if 'Every_N_Lines' in mask_mode:
        out = MergeMask(array, 255 - mask, edge=False)
    else:
        out = MergeMask(array, 255 - mask)
    if save:
        Image.fromarray(out).save(f'input_{mask_mode}.png')
    if dataset in ['FFHQ', 'Places']:
        segm = np.array(Image.open(os.path.join('datasets', dataset, folder, mask_mode, str(number).zfill(6)+'_segm.png')))
        # CLASSES = ['background' ,'skin','nose','eye_g','l_eye','r_eye','l_brow','r_brow','l_ear','r_ear','mouth','u_lip','l_lip','hair','hat','ear_r','neck_l','neck','cloth']
        state = np.random.get_state()
        np.random.seed(44)
        # random palette
        palette = np.random.randint(
            0, 255, size=(segm.max(), 3))
        np.random.set_state(state)
        palette = np.array(palette)
        palette[0] = [0,0,0]

        color_seg = np.zeros((segm.shape[0], segm.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[segm == label, :] = color
    
        if save:
            Image.fromarray(color_seg).save(f'input_segm_{mask_mode}.png')
        return out, color_seg
    else:
        return out, None

if __name__ == '__main__':
    dataset = sys.argv[1]
    mask_mode = sys.argv[2]
    number = sys.argv[3]
    main(dataset, mask_mode, number)
    # track 1
    # python copy_image.py WikiArt Nearest_Neighbor 813 1
    # python copy_image.py WikiArt Expand 983 1
    # python copy_image.py ImageNet ThickStrokes 201 1
    # python copy_image.py FFHQ Completion 305 1
    # python copy_image.py Places Every_N_Lines 182 1

    # track 2
    # python copy_image.py FFHQ Expand 440
    # python copy_image.py Places MediumStrokes 714
    # python copy_image.py Places Nearest_Neighbor 177
    # python copy_image.py FFHQ Completion 308

    # track 1 appendix
    # python copy_image.py FFHQ Completion 561 1
    # python copy_image.py FFHQ Every_N_Lines 571 1
    # python copy_image.py FFHQ Expand 318 1
    # python copy_image.py FFHQ MediumStrokes 276 1
    # python copy_image.py FFHQ Nearest_Neighbor 100 1
    # python copy_image.py FFHQ ThickStrokes 831 1
    # python copy_image.py FFHQ ThinStrokes 540 1

    # python copy_image.py Places Completion 205 1
    # python copy_image.py Places Every_N_Lines 405 1
    # python copy_image.py Places Expand 229 1
    # python copy_image.py Places MediumStrokes 901 1
    # python copy_image.py Places Nearest_Neighbor 582 1
    # python copy_image.py Places ThickStrokes 189 1
    # python copy_image.py Places ThinStrokes 336 1

    # python copy_image.py ImageNet Completion 949 1
    # python copy_image.py ImageNet Every_N_Lines 224 1
    # python copy_image.py ImageNet Expand 854 1
    # python copy_image.py ImageNet MediumStrokes 682 1
    # python copy_image.py ImageNet Nearest_Neighbor 607 1
    # python copy_image.py ImageNet ThickStrokes 341 1
    # python copy_image.py ImageNet ThinStrokes 902 1

    # python copy_image.py WikiArt Completion 249 1
    # python copy_image.py WikiArt Every_N_Lines 343 1
    # python copy_image.py WikiArt Expand 238 1
    # python copy_image.py WikiArt MediumStrokes 385 1
    # python copy_image.py WikiArt Nearest_Neighbor 55 1
    # python copy_image.py WikiArt ThickStrokes 879 1
    # python copy_image.py WikiArt ThinStrokes 416 1

    # track 2 appendix
    # python copy_image.py FFHQ Completion 437 2
    # python copy_image.py FFHQ Every_N_Lines 952 2
    # python copy_image.py FFHQ Expand 243 2
    # python copy_image.py FFHQ MediumStrokes 134 2
    # python copy_image.py FFHQ Nearest_Neighbor 641 2
    # python copy_image.py FFHQ ThickStrokes 826 2
    # python copy_image.py FFHQ ThinStrokes 403 2

    # python copy_image.py Places Completion 296 2
    # python copy_image.py Places Every_N_Lines 417 2
    # python copy_image.py Places Expand 958 2
    # python copy_image.py Places MediumStrokes 223 2
    # python copy_image.py Places Nearest_Neighbor 375 2
    # python copy_image.py Places ThickStrokes 785 2
    # python copy_image.py Places ThinStrokes 179 2
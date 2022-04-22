import numpy as np
import sys
sys.path.append('.')
from mask_generation.utils import MaskGeneration, MergeMask

if __name__ == '__main__':
    from PIL import Image
    import random
    random.seed(1)
    np.random.seed(1)
    image_dir = 'src'
    array = np.array(Image.open(f'{image_dir}/TribunaUffizi.jpg'))
    
    # mode = {
    #     'name': 'ThickStrokes',
    #     'size': 512,
    # }

    # mode = {
    #     'name': 'MediumStrokes',
    #     'size': 512,
    # }

    # mode = {
    #     'name': 'ThinStrokes',
    #     'size': 512,
    # }

    # mode = {
    #     'name': 'Every_N_Lines',
    #     'n': 2,
    #     'direction': 'horizontal'
    # }

    # mode = {
    #     'name': 'Completion',
    #     'ratio': 0.5,
    #     'direction': 'horizontal',
    #     'reverse': False,
    # }

    mode = {
        'name': 'Expand',
        'size': None, # None means half of size
        'direction': 'interior'
    }

    # mode = {
    #     'name': 'Nearest_Neighbor',
    #     'scale': 4,
    #     'upsampling': False,
    # }

    mask_generation = MaskGeneration()
    gt, mask = mask_generation(array, mode, verbose=True)
    # out = np.concatenate((gt, mask), axis=1)
    # mask -> 255: inpaint, 0: keep
    out = MergeMask(array, 255 - mask)
    name = mode.get('name')
    Image.fromarray(out).save(f'{image_dir}/{name}.png')
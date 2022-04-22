import numpy as np
from mask_generation.lama_generation.gen_mask_lama import get_model_from_config, forward_model
import cv2
import random

strokes_configs = [
    'mask_generation/lama_generation/random_medium_256.yaml',
    'mask_generation/lama_generation/random_thick_256.yaml',
    'mask_generation/lama_generation/random_thin_256.yaml',
    'mask_generation/lama_generation/random_medium_512.yaml',
    'mask_generation/lama_generation/random_thick_512.yaml',
    'mask_generation/lama_generation/random_thin_512.yaml',
]

__MASKS__ = ['Every_N_Lines', 'Completion', 'Expand', 'Nearest_Neighbor', 'ThickStrokes', 'MediumStrokes', 'ThinStrokes']

def MergeMask(gt, mask, strc=6, edge=True):
    if mask.ndim == 2:
        mask = mask[:,:,None].repeat(3, axis=2)
    if edge:
        edges = cv2.Canny(mask, 10, 200)
        if strc > 1:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strc, strc))
            edges = cv2.dilate(edges, se, iterations=1)
        edges = np.stack([edges] * 3, axis=2)
        edges[:, :, 0] = edges[:, :, 0] // 255 * 0
        edges[:, :, 1] = edges[:, :, 1] // 255 * 250
        edges[:, :, 2] = edges[:, :, 2] // 255 * 0

    mask_keep0 = 255 - mask
    mask_keep0[:, :, 0] = mask_keep0[:, :, 0] // 255 * 150
    mask_keep0[:, :, 1] = mask_keep0[:, :, 1] // 255 * 150
    mask_keep0[:, :, 2] = mask_keep0[:, :, 2] // 255 * 255

    overlay = mask_keep0.copy()
    if edge:
        np.putmask(overlay, mask_keep0 == 0, edges)

    gt_overlay = gt.copy()
    np.putmask(gt_overlay, overlay != 0, overlay)

    masked = (gt * (1/4) + gt_overlay * (3/4)).astype(np.uint8)

    return masked

class MaskGeneration:
    def __init__(self):
        stroke_shapes = ['medium', 'thick', 'thin']
        stroke_sizes = [256, 512]
        self.stroke_models = {}
        for shape in stroke_shapes:
            self.stroke_models[shape] = {}
            for size in stroke_sizes:
                config = f'mask_generation/lama_generation/random_{shape}_{size}.yaml'
                self.stroke_models[shape][size] = get_model_from_config(config)

    def __call__(self, gt, mode, verbose=False):
        # gt is nympy [h,w,3]
        h, w = gt.shape[:2]
        name = mode.get('name')
        if verbose:
            print('Input:', gt.shape)
        # 255 means inpainting mask, 0 means original image
        if name == 'Every_N_Lines':
            how_many_lines = mode.get('n', 2)
            direction = mode.get('direction', 'vertical')
            mask = np.zeros((h, w), dtype=np.uint8)
            if direction == 'vertical':
                mask[:,::how_many_lines] = 255
            elif direction == 'horizontal':
                mask[::how_many_lines] = 255
            else:
                raise TypeError("Please select a valid direction")

        elif name == 'Completion':
            ratio =  mode.get('ratio', 0.5)
            assert 0 < ratio <= 1, "Ratio must be between 0 and 1" 
            reverse =  mode.get('reverse', False)
            direction = mode.get('direction', 'vertical')
            mask = np.zeros((h, w), dtype=np.uint8)
            if direction == 'vertical':
                mask[: int(h * ratio)] = 255
            elif direction == 'horizontal':
                mask[:, : int(w * ratio)] = 255
            else:
                raise TypeError(f"Please select a valid direction. {direction} not valid")
            if reverse:
                mask = 255 - mask

        elif name == 'Expand':
            size = mode.get('size')
            if size is None:
                size = min(gt.shape[:-1]) // 2
            elif isinstance(size, str) and size == 'random':
                min_size = min(gt.shape[:-1])
                size = random.choice([min_size//4, min_size//3, min_size//2])
                
            size //= 2
            direction = mode.get('direction', 'interior')
            reverse =  mode.get('reverse', False)
            if direction == 'interior':
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[
                        h // 2 - size: h // 2 + size,
                        w // 2 - size: w // 2 + size
                    ] = 0
            elif direction == 'exterior':
                mask = np.ones((h+size*2, w+size*2)) * 255
                mask[size:-size, size:-size] = 0
                if gt.ndim == 3:
                    new_gt = np.zeros((h+size*2, w+size*2, gt.shape[-1]))
                else:
                    new_gt = np.zeros((h+size*2, w+size*2))
                new_gt[size:-size, size:-size] = gt
                gt = new_gt
            else:
                raise TypeError(f"Please select a valid direction. {direction} not valid")
            if reverse:
                mask = 255 - mask

        elif name == 'Nearest_Neighbor':
            scale = mode.get('scale', 2)
            upsampling = mode.get('upsampling', True)
            assert isinstance(scale, int), f"scale={scale} must be an integer"
            
            if upsampling:
                if gt.ndim == 3:
                    gt_up = np.zeros((h*scale, w*scale, gt.shape[-1]), dtype=np.uint8)
                else:
                    gt_up = np.zeros((h*scale, w*scale), dtype=np.uint8)
                gt_up[::scale, ::scale] = gt
                gt = gt_up
                mask = np.ones((h*scale, w*scale), dtype=np.uint8) * 255
                mask[::scale, ::scale] = 0
            else:
                gt_up = np.zeros_like(gt)
                gt_up[::scale, ::scale] = gt[::scale, ::scale]
                gt = gt_up
                mask = np.ones((h, w), dtype=np.uint8) * 255
                mask[::scale, ::scale] = 0

        elif name == 'ThickStrokes':
            shape = 'thick'
            size = mode.get('size', 512)
            model = self.stroke_models[shape][size]
            config = f'mask_generation/lama_generation/random_{shape}_{size}.yaml'
            mask = forward_model(config, model, gt)

        elif name == 'MediumStrokes':
            shape = 'medium'
            size = mode.get('size', 512)
            model = self.stroke_models[shape][size]
            config = f'mask_generation/lama_generation/random_{shape}_{size}.yaml'
            mask = forward_model(config, model, gt)

        elif name == 'ThinStrokes':
            shape = 'thin'
            size = mode.get('size', 512)
            model = self.stroke_models[shape][size]
            config = f'mask_generation/lama_generation/random_{shape}_{size}.yaml'
            mask = forward_model(config, model, gt)

        else:
            raise NotImplementedError(mode)

        if verbose:
            print('Output GT       :', gt.shape)
            print('Output Mask     :', mask.shape)
            print('Inpainting Ratio: {:0.2f}%'.format(mask.mean()*100/255))
            # 255 means inpainting mask, 0 means original image
        if gt.ndim == 3:
            gt = gt * (mask[:,:,None]<1)
        else:
            gt = gt * (mask<1)
        return gt, mask

def RandomAttribute(mask, size):
    if mask in ['ThickStrokes', 'MediumStrokes', 'ThinStrokes']:
        mode = {
            'name': mask,
            'size': size,
        }

    elif mask == 'Every_N_Lines':
        mode = {
            'name': 'Every_N_Lines',
            'n': random.choice([2,3,4,5]),
            'direction': random.choice(['horizontal', 'vertical'])
        }

    elif mask == 'Completion':
        mode = {
            'name': 'Completion',
            'ratio': random.uniform(0.2, 0.8),
            'direction': random.choice(['horizontal', 'vertical']),
            'reverse': random.choice([True, False]),
        }

    elif mask == 'Expand':
        mode = {
            'name': 'Expand',
            'size': 'random', # None means half of size
            'direction': 'interior'
        }

    elif mask == 'Nearest_Neighbor':
        mode = {
            'name': 'Nearest_Neighbor',
            'scale': random.choice([2,3,4]),
            'upsampling': False,
        }
    
    else:
        raise TypeError(f"Select a valid mask [{mask}]")
    
    return mode
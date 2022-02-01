#!/usr/bin/env python3

import glob
import os
import traceback

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

from mask_generation.lama_generation.MixedMaskGenerator import MixedMaskGenerator
import tempfile
import yaml
from easydict import EasyDict as edict
from enum import Enum

class SmallMode(Enum):
    DROP = "drop"
    UPSCALE = "upscale"

def load_yaml(path):
    with open(path, 'r') as f:
        return edict(yaml.safe_load(f))

def propose_random_square_crop(mask, min_overlap=0.5):
    height, width = mask.shape
    mask_ys, mask_xs = np.where(mask > 0.5)  # mask==0 is known fragment and mask==1 is missing

    if height < width:
        crop_size = height
        obj_left, obj_right = mask_xs.min(), mask_xs.max()
        obj_width = obj_right - obj_left
        left_border = max(0, min(width - crop_size - 1, obj_left + obj_width * min_overlap - crop_size))
        right_border = max(left_border + 1, min(width - crop_size, obj_left + obj_width * min_overlap))
        start_x = np.random.randint(left_border, right_border)
        return start_x, 0, start_x + crop_size, height
    else:
        crop_size = width
        obj_top, obj_bottom = mask_ys.min(), mask_ys.max()
        obj_height = obj_bottom - obj_top
        top_border = max(0, min(height - crop_size - 1, obj_top + obj_height * min_overlap - crop_size))
        bottom_border = max(top_border + 1, min(height - crop_size, obj_top + obj_height * min_overlap))
        start_y = np.random.randint(top_border, bottom_border)
        return 0, start_y, width, start_y + crop_size

class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, indir, outdir, config):
    if config.generator_kind == 'random':
        variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                              variants_n=variants_n)
    else:
        raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < config.cropping.out_min_size:
                handle_small_mode = SmallMode(config.cropping.handle_small_mode)
                if handle_small_mode == SmallMode.DROP:
                    continue
                elif handle_small_mode == SmallMode.UPSCALE:
                    factor = config.cropping.out_min_size / min(image.size)
                    out_size = (np.array(image.size) * factor).round().astype('uint32')
                    image = image.resize(out_size, resample=Image.BICUBIC)
            else:
                factor = config.cropping.out_min_size / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                else:
                    cur_image = image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((cur_image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                cur_image.save(cur_basename + '.png')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


def main(args):

    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    config = load_yaml(args.config)

    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))
    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )

def get_model_from_config(config):
    config = load_yaml(config)
    if config.generator_kind == 'random':
        variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                              variants_n=variants_n)
    else:
        raise ValueError(f'Unexpected generator kind: {config.generator_kind}')
    return mask_generator

def process_image_from_model(mask_generator, infile, outfile, config):
    # if config.generator_kind == 'random':
    #     variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
    #     mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
    #                                           variants_n=variants_n)
    # else:
    #     raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)
    # import ipdb; ipdb.set_trace()
    try:
        image = Image.open(infile).convert('RGB')

        # scale input image to output resolution and filter smaller images
        # if min(image.size) < config.cropping.out_min_size:
        #     handle_small_mode = SmallMode(config.cropping.handle_small_mode)
        #     if handle_small_mode == SmallMode.DROP:
        #         raise ValueError(f"Image too small. {image.size}.")
        #     elif handle_small_mode == SmallMode.UPSCALE:
        #         factor = config.cropping.out_min_size / min(image.size)
        #         out_size = (np.array(image.size) * factor).round().astype('uint32')
        #         image = image.resize(out_size, resample=Image.BICUBIC)
        # else:
        #     factor = config.cropping.out_min_size / min(image.size)
        #     out_size = (np.array(image.size) * factor).round().astype('uint32')
        #     image = image.resize(out_size, resample=Image.BICUBIC)

        # generate and select masks
        src_masks = mask_generator.get_masks(image)

        filtered_image_mask_pairs = []
        for cur_mask in src_masks:
            cur_image = image

            if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                continue

            filtered_image_mask_pairs.append((cur_image, cur_mask))

        mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                        size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                        replace=False)

        # crop masks; save masks together with input image
        # mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
        for i, idx in enumerate(mask_indices):
            cur_image, cur_mask = filtered_image_mask_pairs[idx]
            cur_basename = outfile # mask_basename + f'_crop{i:03d}'
            Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                            mode='L').save(cur_basename) # + f'_mask{i:03d}.png')
            cur_image.save(cur_basename) # + '.png')
    except KeyboardInterrupt:
        return
    except Exception as ex:
        print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')

def forward_model(config, model, gt_numpy):
    config = load_yaml(config)
    tmp_file = tempfile.NamedTemporaryFile(suffix='.png')
    Image.fromarray(gt_numpy).save(tmp_file)
    tmp_file_out = tempfile.NamedTemporaryFile(suffix='.png')
    while True:
        process_image_from_model(model, tmp_file, tmp_file_out, config)
        try:
            mask = np.array(Image.open(tmp_file_out))
            break
        except:
            pass
    return mask


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--n-jobs', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())

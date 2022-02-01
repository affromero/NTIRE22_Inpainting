import argparse
import numpy as np
import PIL.Image
from glob import glob
from dnnlib import tflib
from training import misc
import os
from tqdm import tqdm

def generate(Gs, image, mask, output, truncation):
    real = np.asarray(PIL.Image.open(image)).transpose([2, 0, 1])
    real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
    mask = np.asarray(PIL.Image.open(mask).convert('1'), dtype=np.float32)[np.newaxis]
    import ipdb; ipdb.set_trace()
    latent = np.random.randn(1, *Gs.input_shape[1:])
    fake = Gs.run(latent, None, real[np.newaxis], mask[np.newaxis], truncation_psi=truncation)[0]
    fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
    fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
    fake = PIL.Image.fromarray(fake)
    fake.save(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', required=True)
    # parser.add_argument('-i', '--image', help='Original image path', required=True)
    # parser.add_argument('-m', '--mask', help='Mask path', required=True)
    # parser.add_argument('-o', '--output', help='Output (inpainted) image path', required=True)
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-s', '--save-dir', default='images')    
    parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)

    args = parser.parse_args()
    img_list = sorted([i for i in glob(os.path.join(args.data_dir, '*.png')) if 'mask' not in i])
    mask_list = sorted([i for i in glob(os.path.join(args.data_dir, '*.png')) if 'mask' in i])
    os.makedirs(args.save_dir, exist_ok=True)
    tflib.init_tf()
    _, _, Gs = misc.load_pkl(args.checkpoint)
    for i,m in tqdm(zip(img_list, mask_list), total=len(img_list)):
        out = os.path.join(args.save_dir, os.path.basename(i))
        generate(Gs, i, m, out, args.truncation)

if __name__ == "__main__":
    main()

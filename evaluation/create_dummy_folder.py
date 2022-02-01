import os
from glob import glob
from shutil import copy
from tqdm import tqdm 
dataset = 'Places'
in_dir = f'datasets/{dataset}/val'
out_dir = 'dummy_validation/{dataset}'
mask_types = os.listdir(in_dir)
for mask in tqdm(mask_types):
    mask_dir = os.path.join(out_dir, mask)
    os.makedirs(mask_dir, exist_ok=True)
    files = glob(os.path.join(in_dir, mask, '*png'))
    files = [i for i in files if '_' not in i]
    for filename in tqdm(files, leave=False):
        name = os.path.basename(filename)
        integer = int(name.split('.')[0])
        if integer % 10 == 0:
            new_file = os.path.join(mask_dir, name)
            copy(filename, new_file)
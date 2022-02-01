import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    opt = TestOptions().parse()
    model = create_model(opt) # .to(device)
    model.netEN.module.load_state_dict(torch.load(os.path.join(opt.checkpoint_folder, "EN.pkl")))
    model.netDE.module.load_state_dict(torch.load(os.path.join(opt.checkpoint_folder, "DE.pkl")))
    model.netMEDFE.module.load_state_dict(torch.load(os.path.join(opt.checkpoint_folder, "MEDEF.pkl")))
    results_dir = opt.out_dir
    if not os.path.exists( results_dir):
        os.mkdir(results_dir)

    img_paths = glob('{:s}/*png'.format(opt.in_dir))
    png_paths = sorted([i for i in img_paths if 'mask' not in i])
    mask_paths = sorted([i for i in img_paths if 'mask' in i])    

    st_root = os.path.abspath('tmp_structure')
    os.system(f'rm -rf {st_root}')
    pwd = os.getcwd()
    os.chdir(os.path.join('data', 'Matlab'))
    matlab_command = f"generate_structure_images('{opt.in_dir}', '{st_root}'); exit"
    print('executing matlab command:', matlab_command)
    os.system(f'matlab -nodisplay -nodesktop -r "{matlab_command}"')
    os.chdir(pwd)    
    st_root = sorted([i for i in glob(st_root+'/*png') if 'mask' not in i])
    # mask_paths = glob('{:s}/*'.format(opt.mask_root))
    # de_paths = glob('{:s}/*'.format(opt.de_root))
    # st_path = glob('{:s}/*'.format(opt.st_root))
    # image_len = len(de_paths )
    for i, m, s in tqdm(zip(png_paths, mask_paths, st_root), total=len(png_paths)):
        mask = Image.open(m).convert("RGB")
        detail = Image.open(i).convert("RGB")
        structure = Image.open(s).convert("RGB")

        mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)
        structure = mask*detail

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        filename = os.path.basename(i)
        output.save(f"{results_dir}/{filename}")

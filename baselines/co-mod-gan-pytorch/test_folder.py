import argparse
import numpy as np
import torch
from co_mod_gan import Generator
from PIL import Image
from glob import glob 
import os
from tqdm import tqdm
import torch.nn.functional as F

def check_padding(x, pad=512):
    diffY = 0 if x.size(-2) % pad == 0 else pad - x.size(-2) % pad
    diffX = 0 if x.size(-1) % pad == 0 else pad - x.size(-1) % pad	
    padding = [diffX // 2, diffX - diffX // 2, 
        diffY // 2, diffY - diffY // 2]
    x = F.pad(x, padding) 
    return x, padding  

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', required=True)
parser.add_argument('-d', '--data_dir', help='Original image path', required=True)
parser.add_argument('-s', '--save_dir', help='Output (inpainted) image path', required=True)
parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)
parser.add_argument('--device', help='cpu|cuda', default='cuda')

args = parser.parse_args()

assert args.truncation is None

device = torch.device(args.device)

net = Generator()
net.load_state_dict(torch.load(args.checkpoint))
net.eval()

net = net.to(device)

img_list = sorted([i for i in glob(os.path.join(args.data_dir, '*.png')) if 'mask' not in i and 'segm' not in i])
mask_list = sorted([i for i in glob(os.path.join(args.data_dir, '*.png')) if 'mask' in i])
os.makedirs(args.save_dir, exist_ok=True)

for i,m in tqdm(zip(img_list, mask_list), total=len(img_list)):
    out = os.path.join(args.save_dir, os.path.basename(i))

    real = np.asarray(Image.open(i).convert('RGB')).transpose([2, 0, 1])/255.0
    masks = 1 - np.asarray(Image.open(m).convert('1'), dtype=np.float32)

    images = torch.Tensor(real.copy())[None,...]*2-1
    masks = torch.Tensor(masks)[None,None,...].float()
    masks = (masks>0).float()
    latents_in = torch.randn(1, 512)

    images = images.to(device)
    masks = masks.to(device)
    latents_in = latents_in.to(device)

    # import ipdb; ipdb.set_trace()
    images, padding = check_padding(images)
    masks = check_padding(masks)[0] 
    if images.size(-2) > 512 or masks.size(-1) > 512:
        org_size = images.size()
        images = images.unfold(2,512,512).unfold(3,512,512)
        masks = masks.unfold(2,512,512).unfold(3,512,512)

        result = torch.zeros_like(images)
        for i in range(images.size(2)):
            for j in range(images.size(3)):
                with torch.no_grad():
                    try:
                        result[:,:,i,j,:,:] = net(images[:,:,i,j,:,:], masks[:,:,i,j,:,:], [latents_in], truncation=args.truncation)
                    except RuntimeError:
                        pass
        
        # import ipdb; ipdb.set_trace()
        result = result.permute(0, 1, 2, 4, 3, 5).contiguous()
        result = result.view(*org_size)

    else:
        result = net(images, masks, [latents_in], truncation=args.truncation)
    result = F.pad(result, [-i for i in padding])
    result = result.detach().cpu().numpy()
    result = (result+1)/2
    result = (result[0].transpose((1,2,0)))*255
    Image.fromarray(result.clip(0,255).astype(np.uint8)).save(out)
    # import ipdb; ipdb.set_trace()

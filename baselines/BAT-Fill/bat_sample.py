import os
import numpy as np
import torch
import torch.nn.functional as F
import time
from glob import glob
from PIL import Image
from tqdm import tqdm
import argparse
from models.bat_model import GPT, GPTConfig
from models.networks.generator import UpsamplerGenerator

parser = argparse.ArgumentParser(description='PyTorch Template')

parser.add_argument('--num_sample', type=int, default=1, help='input batch size')
parser.add_argument('--tran_model', type=str, default='celebahq_bat_pretrain', help='name of BAT model')
parser.add_argument('--up_model', type=str, default='celebahq_up_pretrain', help='name of upsampler model')
parser.add_argument('--input_dir', type=str, help='dir of input images, png foramt is hardcoded in line 121, please modify if needed.')
parser.add_argument('--mask_dir', type=str, help='dir of masks, filename should match with input images')
parser.add_argument('--save_dir', type=str, help='dir for saving results')

args = parser.parse_args()
def imread_torch(img_path, mask_path, size=256):
    img = torch.from_numpy(np.array(Image.open(img_path).resize([size,size], Image.BICUBIC)))
    img = img.permute(2,0,1)/127.5 - 1.
    mask = torch.from_numpy(np.array(Image.open(mask_path).resize([size,size], Image.NEAREST)))
    mask = (mask > 0).float()
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    img, mask = img.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0)
    masked_img = img*(1-mask) + mask
    return img, mask, masked_img

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


cluster = torch.from_numpy(np.load('./kmeans_centers.npy'))
def color_quantize(x):
        # x (3,32,32)
        xpt = x.float().permute(1,2,0).contiguous().view(-1, 3)
        ix = ((xpt[:, None, :] - cluster[None, :, :])**2).sum(-1).argmin(1)
        return ix
def dequantize(ix, size=32):
    return (127.5 * (cluster[ix] + 1.0)).view(size, size, 3).numpy().astype(np.uint8)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def masked_sample(model, x, masks, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    bs, t = x.shape
    pred = x[:,~masks[0].bool()][:,:-1]
    x_out = x.clone()
    mask_ids = torch.nonzero(1 - masks[0].flatten())
    model.eval()
    for step_n, mask_id in enumerate(mask_ids):
        logits, _ = model(x, pred, masks)
        logits = logits[:, step_n, :] / temperature
        logits = logits.squeeze(1)
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        if step_n < len(mask_ids)-1:
            pred[:, [step_n]] = ix
        x_out[:, [mask_id]] = ix
    return x_out

if __name__=="__main__":
    num_sample = args.num_sample
    tran_model = args.tran_model
    up_model = args.up_model
    input_dir = args.input_dir
    mask_dir = args.input_dir # args.mask_dir
    save_dir = args.save_dir
    block_size = 32 ** 2
    vocab_size = 512
    ckpt_path = './checkpoints/{}/latest_tran.pth'.format(tran_model)
    mconf = GPTConfig(vocab_size, block_size*2,
                    embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                    n_layer=24, n_head=8, n_embd=512)
    model = GPT(mconf)
    weights = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(weights)
    model = model.cuda()
    class Options():
        netG = 'Upsampler'
        ngf = 64
        norm_G = 'spectralspadeposition3x3'
        resnet_n_blocks = 6
        use_attention = True
        input_nc = 4
        gpu_ids = [0]
        semantic_nc = 4
    
    save_path = './checkpoints/{}/latest_net_G.pth'.format(up_model)
    opt = Options()
    netG = UpsamplerGenerator(opt)
    weights = torch.load(save_path, map_location='cpu')
    netG.load_state_dict(weights)
    netG = netG.cuda()
    netG.eval()
    create_dir(save_dir)
    png_paths = sorted([i for i in glob(input_dir+'/*.png') if 'mask' not in i])
    mask_paths = sorted([i for i in glob(input_dir+'/*.png') if 'mask' in i])
    # test 100 for quick eval
    for p, m in tqdm(zip(png_paths, mask_paths), total=len(png_paths)):
        img, mask, masked_img = imread_torch(p, m, 256)
        _, mask_32, img_32 = imread_torch(p, m, 32)
        quant_imgs = color_quantize(img_32[0]).unsqueeze(0)
        mask_32 = 1.0 - mask_32.view(1,-1)

        quant_gens = masked_sample(model, quant_imgs.repeat(num_sample,1).cuda(), mask_32.repeat(num_sample,1).cuda(), sample=True, top_k=50)
        masked_img = torch.cat([masked_img, mask], 1)
        for i in range(num_sample):
            sample = dequantize(quant_gens[i])
            sample_tensor =  torch.from_numpy(sample).permute(2,0,1).unsqueeze(0).float()
            sample_tensor = sample_tensor/127.5 -1
            
            _, sample_up = netG([masked_img.cuda(), sample_tensor.cuda()])
            sample_up = sample_up.cpu() * mask + masked_img[:,:3] * (1-mask)
            sample_up = sample_up[0].permute(1,2,0).detach().numpy()
            sample_up = ((sample_up+1)*127.5).astype(np.uint8)
            if num_sample == 1:
                Image.fromarray(sample_up).save(os.path.join(save_dir, p.split('/')[-1]))
            else:
                Image.fromarray(sample_up).save(os.path.join(save_dir, p.split('/')[-1].replace('.png','_{}.png'.format(i))))
#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from torchvision.transforms.functional import center_crop

LOGGER = logging.getLogger(__name__)

def check_padding(x):
    diffY = 0 if x.size(-2) % 8 == 0 else 8 - x.size(-2) % 8
    diffX = 0 if x.size(-1) % 8 == 0 else 8 - x.size(-1) % 8	
    padding = [diffX // 2, diffX - diffX // 2, 
        diffY // 2, diffY - diffY // 2]
    x = F.pad(x, padding) 
    return x, padding  

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        # import ipdb; ipdb.set_trace()
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                mask_fname = mask_fname.replace('_mask.', '.')
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)

                if 'FFHQ' in mask_fname:
                    # ffhq was trained for 256
                    batch['image'] = F.interpolate(batch['image'], (256,256), mode='bilinear', align_corners=False)
                    if 'Nearest' in mask_fname or 'Every_N' in mask_fname:
                        batch['mask'] = center_crop(batch['mask'], (256,256))
                    else:
                        batch['mask'] = F.interpolate(batch['mask'], (256,256), mode='nearest')
                # import ipdb; ipdb.set_trace()
                batch['image'], padding = check_padding(batch['image'])  
                # if np.sum(padding):
                #     import ipdb; ipdb.set_trace()
                batch['mask'], _ = check_padding(batch['mask'])  
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch[predict_config.out_key]
                assert cur_res.size() == batch['image'].size(), "output image must be same size as input"
                cur_res = F.pad(cur_res, [-i for i in padding]) 
                # cur_res = cur_res[:,:,padding[2]:cur_res.size(2)-padding[3], padding[0]:cur_res.size(3)-padding[1]]
                cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                cv2.imwrite(cur_out_fname, cur_res)
                # import ipdb; ipdb.set_trace()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()

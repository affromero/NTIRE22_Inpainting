"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    tokens = 0

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.cluster = torch.from_numpy(np.load('./kmeans_centers.npy'))
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    
    def color_quantize(self, x):
        # x (3,32,32)
        xpt = x.float().permute(1,2,0).contiguous().view(-1, 3)
        ix = ((xpt[:, None, :] - self.cluster[None, :, :])**2).sum(-1).argmin(1)
        return ix
    def dequantize(self, ix, size=32):
        return (127.5 * (self.cluster[ix] + 1.0)).view(size, size, 3).numpy().astype(np.uint8)

    def preprocess_input(self, data):
        image = data['image']
        bs, c, h, w = image.shape
        # use same masks in the same batch
        masks = data['mask'][:1].repeat(bs,1,1,1)
        masked_img = image * (1-masks) + masks
        quant_masked_img = []
        quant_img = []
        for i in range(bs):
            quant_masked_img.append(self.color_quantize(masked_img[i]))
            quant_img.append(self.color_quantize(image[i]))
        quant_masked_img = torch.stack(quant_masked_img, 0) # (b, L)
        quant_img = torch.stack(quant_img, 0)
        masks = 1.0 - masks.view(bs, -1)
        # permute mask and extent mask for predicted part
        masks_ext = torch.cat([masks[:,masks[0].bool()], masks[:,~masks[0].bool()], torch.zeros(bs,1), torch.ones(bs,h*w-1)],1)
        return quant_masked_img.to(self.device), masks.to(self.device), quant_img[:,~masks[0].bool()].to(self.device)

    def load_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if os.path.exists(self.config.ckpt_path):
            weights = torch.load(self.config.ckpt_path, map_location='cpu')
            raw_model.load_state_dict(weights)
            print("loading ", self.config.ckpt_path)
        return raw_model

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print("saving ", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, opt, dataset):
        model, config = self.model, self.config
        raw_model = self.load_checkpoint()
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):

            is_train = split == 'train'
            if 'places' in opt.dataset_name and is_train:
                #reinit dataset, shuffle and select from large dataset
                self.train_dataset = dataset.create_dataloader(opt) 
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, data in pbar:

                # place data on the correct device
                x, mask, y = self.preprocess_input(data)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y[:,:-1], mask, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                  
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: ", test_loss)
                return test_loss
            else:
                train_loss = float(np.mean(losses))
                return train_loss 
            
        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        self.log_name = os.path.join(os.path.dirname(self.config.ckpt_path), 'tran_loss_log.txt')
        self.iter_record_path = os.path.join(os.path.dirname(self.config.ckpt_path), 'tran_iter.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        try:
            self.first_epoch, self.tokens = np.loadtxt(
                    self.iter_record_path, delimiter=',', dtype=int)
            print('Resuming from epoch %d at token %d' % (self.first_epoch, self.tokens))
        except Exception as e:
            print(e)
            self.first_epoch = 1
            self.tokens = 0
            print('Could not load iteration record at %s. Starting from beginning.' %
                      self.iter_record_path)
        for epoch in range(self.first_epoch-1, config.max_epochs):

            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
            message = 'Epoch {}, Train loss: {}, Test loss: {}'.format(epoch+1, train_loss, test_loss)
            print(message)
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)

            np.savetxt(self.iter_record_path, (epoch+1, self.tokens),
                   delimiter=',', fmt='%d')
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

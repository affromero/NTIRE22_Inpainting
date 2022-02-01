"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
from options.train_options import TrainOptions
import data
from trainers.bat_trainer import Trainer, TrainerConfig
from models.bat_model import set_seed, GPT, GPTConfig
set_seed(42)

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
opt.phase = 'test'
val_dataloader = data.create_dataloader(opt)
opt.phase = 'train'

block_size = opt.load_size ** 2
vocab_size = 512
mconf = GPTConfig(vocab_size, block_size*2,
                  embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                  n_layer=24, n_head=8, n_embd=512, ar_bert_loss=True)

model = GPT(mconf)

tokens_per_epoch = len(dataloader) * block_size * opt.batchSize
train_epochs = 150

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, learning_rate=3e-4,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path=os.path.join(opt.checkpoints_dir, opt.name, 'latest_tran.pth'),
                      num_workers=4)
trainer = Trainer(model, dataloader, val_dataloader, tconf)
trainer.train(opt, data)

print('Training was successfully finished.')

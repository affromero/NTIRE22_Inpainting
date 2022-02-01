"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
import torch
import numpy as np
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch.nn.functional as F

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            if i % 500 == 0:
                trainer.run_generator_one_step(data_i, True)
            else:
                trainer.run_generator_one_step(data_i)
        # train discriminator
        if i % 500 == 0:
            trainer.run_discriminator_one_step(data_i, True)
        else:
            trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            sample = trainer.get_latest_generated()[0] # either 32x32 gt or predicted samples as input
            sample = F.interpolate(sample, scale_factor=8)
            visuals = OrderedDict([('input', data_i['masked_img']),
                                   ('synthesized_image', trainer.get_latest_generated()[-1]),
                                   ('sample', sample),
                                   ('real_image', data_i['image'])])
            combine_image = torch.cat([data_i['masked_img'][:,:3].cpu(),
                trainer.get_latest_generated()[-1].cpu(),
                sample.cpu(),
                data_i['image'].cpu()], dim=3)
            visual_combine = OrderedDict([('visualization', combine_image)])
            visualizer.display_current_results(visual_combine, epoch, iter_counter.total_steps_so_far)
            # visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
    
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
print('Training was successfully finished.')

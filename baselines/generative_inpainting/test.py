import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
from glob import glob
from inpaint_model import InpaintCAModel
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output', type=str,
                    help='Where to write output.')
parser.add_argument('--image_height', default=256, type=str,
                    help='Where to write output.')                    
parser.add_argument('--image_width', default=256, type=str,
                    help='Where to write output.')                                        
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    args, unknown = parser.parse_known_args()
    sess_config = tf.ConfigProto()                                                                                                                                                                                                            
    sess_config.gpu_options.allow_growth = True                                                                                                                                                                                               
    sess = tf.Session(config=sess_config)                                                                                                                                                                                                     
                                                                                                                                                                                                                                              
    model = InpaintCAModel()                                                                                                                                                                                                                  
    input_image_ph = tf.placeholder(                                                                                                                                                                                                          
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))                                                                                                                                                                      
    output = model.build_server_graph(FLAGS, input_image_ph)                                                                                                                                                                                         
    output = (output + 1.) * 127.5                                                                                                                                                                                                            
    output = tf.reverse(output, [-1])                                                                                                                                                                                                         
    output = tf.saturate_cast(output, tf.uint8)                                                                                                                                                                                               
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)                                                                                                                                                                              
    assign_ops = []                                                                                                                                                                                                                           
    for var in vars_list:                                                                                                                                                                                                                     
        vname = var.name                                                                                                                                                                                                                      
        from_name = vname                                                                                                                                                                                                                     
        var_value = tf.contrib.framework.load_variable(                                                                                                                                                                                       
            args.checkpoint_dir, from_name)                                                                                                                                                                                                   
        assign_ops.append(tf.assign(var, var_value))                                                                                                                                                                                          
    sess.run(assign_ops)                                                                                                                                                                                                                      
    print('Model loaded.')                                                                                                                                                                                                                    
                                                                                                                                                                                                                                              
    images = sorted([i for i in glob(os.path.join(args.image, '*.png')) if 'mask' not in i])
    masks = sorted([i for i in glob(os.path.join(args.image, '*.png')) if 'mask' in i])
    os.makedirs(args.output, exist_ok=True)
    t = time.time()                                                                                                                                                                                                           
    for _image, _mask in tqdm(zip(images, masks), total=len(images)):                                                                                                                                                                                                              
        image = cv2.imread(_image)
        mask = cv2.imread(_mask)                                                                                                                                                                                                                                                                                                                                                                                                            
        image = cv2.resize(image, (args.image_width, args.image_height))                                                                                                                                                                      
        mask = cv2.resize(mask, (args.image_width, args.image_height))                                                                                                                                                                        
        # cv2.imwrite(out, image*(1-mask/255.) + mask)                                                                                                                                                                                        
        # # continue                                                                                                                                                                                                                          
        # image = np.zeros((128, 256, 3))                                                                                                                                                                                                     
        # mask = np.zeros((128, 256, 3))                                                                                                                                                                                                      
                                                                                                                                                                                                                                              
        assert image.shape == mask.shape                                                                                                                                                                                                      
                                                                                                                                                                                                                                              
        h, w, _ = image.shape                                                                                                                                                                                                                 
        grid = 4                                                                                                                                                                                                                              
        image = image[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                        
        mask = mask[:h//grid*grid, :w//grid*grid, :]                                                                                                                                                                                          
        print('Shape of image: {}'.format(image.shape))                                                                                                                                                                                       
                                                                                                                                                                                                                                              
        image = np.expand_dims(image, 0)                                                                                                                                                                                                      
        mask = np.expand_dims(mask, 0)                                                                                                                                                                                                        
        input_image = np.concatenate([image, mask], axis=2)                                                                                                                                                                                   
                                                                                                                                                                                                                                              
        # load pretrained model                                                                                                                                                                                                               
        result = sess.run(output, feed_dict={input_image_ph: input_image})          
        out = os.path.join(args.output, os.path.basename(_image))                                                                                                                                                          
        print('Processed: {}'.format(out))                                                                                                                                                                                                    
        cv2.imwrite(out, result[0][:, :, ::-1])                                                                                                                                                                                               
                                                                                                                                                                                                                                              
    print('Time total: {}'.format(time.time() - t)) 


    # FLAGS = ng.Config('inpaint.yml')
    # # ng.get_gpus(1)
    # args, unknown = parser.parse_known_args()

    # model = InpaintCAModel()
    # images = sorted([i for i in glob(os.path.join(args.image, '*.png')) if 'mask' not in i])
    # masks = sorted([i for i in glob(os.path.join(args.image, '*.png')) if 'mask' in i])
    # os.makedirs(args.output, exist_ok=True)
    # for _image, _mask in tqdm(zip(images, masks), total=len(images)):
    #     image = cv2.imread(_image)
    #     mask = cv2.imread(_mask)
    #     # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    #     assert image.shape == mask.shape

    #     h, w, _ = image.shape
    #     grid = 8
    #     image = image[:h//grid*grid, :w//grid*grid, :]
    #     mask = mask[:h//grid*grid, :w//grid*grid, :]
    #     print('Shape of image: {}'.format(image.shape))

    #     image = np.expand_dims(image, 0)
    #     mask = np.expand_dims(mask, 0)
    #     input_image = np.concatenate([image, mask], axis=2)

    #     sess_config = tf.ConfigProto()
    #     sess_config.gpu_options.allow_growth = True
    #     with tf.Session(config=sess_config) as sess:
    #         input_image = tf.constant(input_image, dtype=tf.float32)
    #         output = model.build_server_graph(FLAGS, input_image, reuse=tf.AUTO_REUSE)
    #         output = (output + 1.) * 127.5
    #         output = tf.reverse(output, [-1])
    #         output = tf.saturate_cast(output, tf.uint8)
    #         # load pretrained model
    #         vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #         assign_ops = []
    #         for var in vars_list:
    #             vname = var.name
    #             from_name = vname
    #             var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
    #             assign_ops.append(tf.assign(var, var_value))
    #         sess.run(assign_ops)
    #         print('Model loaded.')
    #         result = sess.run(output)
    #         filename = os.path.join(args.output, os.path.basename(_image))
    #         cv2.imwrite(filename, result[0][:, :, ::-1])

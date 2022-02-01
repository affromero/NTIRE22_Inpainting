
if __name__=='__main__':
    import os
    from glob import glob
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=['ict', 'generative_inpainting', 'batfill', 'dsi', 'aot', 'medfe'])
    parser.add_argument("--root_tar", type=str, default='/cluster/work/cvl/gudiff/baseline_datasets')
    parser.add_argument("--image_folder", type=str, default='images_input')
    parser.add_argument("--mask_folder", type=str, default='mask_input')
    # parser.add_argument("--tmp_dir", type=str, default='output')
    parser.add_argument("--single_tar", action='store_true')
    parser.add_argument("--mask_mode", type=str, default='drop', choices=['keep', 'drop'], help='Whether to keep pixels with one or to remove them')
    args=parser.parse_args()

    if args.method == 'ict':
        folder_method = 'ICT'
    elif args.method == 'generative_inpainting':
        folder_method = 'generative_inpainting'
    elif args.method == 'batfill':
        folder_method = 'BAT-Fill'
    elif args.method == 'dsi':
        folder_method = 'Diverse-Structure-Inpainting'   
    elif args.method == 'aot':
        folder_method = 'AOT-GAN-for-Inpainting'  
    elif args.method == 'medfe':
        folder_method = 'Rethinking-Inpainting-MEDFE'                        

    tmp_dir = os.environ["TMPDIR"]
    os.system(f'rm -rf {tmp_dir}/inpainting_baselines')
    os.chdir(tmp_dir)
    os.system('git clone https://github.com/affromero/inpainting_baselines.git')
    os.chdir(f'./inpainting_baselines/{folder_method}')

    root = args.root_tar
    method = args.method
    image_folder = args.image_folder
    mask_folder = args.mask_folder
    if args.single_tar:
        images = [args.root_tar]
        masks = [args.root_tar.replace('gt_256', f'gt_{args.mask_mode}_mask_256')]
        # creating final dir
        out_dir_tar = os.path.join(os.path.dirname(root), method)
        
    else:
        # images = sorted([i for i in glob(os.path.join(args.root_tar, '*.tar')) if 'mask' not in i])
        # masks = sorted([i for i in glob(os.path.join(args.root_tar, '*.tar')) if 'mask' in i and args.mask_mode in i])
        images = sorted([i for i in glob(os.path.join(args.root_tar, '*sample*.tar')) if 'mask' not in i])
        masks = sorted([i for i in glob(os.path.join(args.root_tar, '*sample*.tar')) if 'mask' in i and args.mask_mode in i])        
        # creating final dir
        out_dir_tar = os.path.join(root, method)
    os.makedirs(out_dir_tar, exist_ok=True)        
    pwd = os.getcwd()

    for i, m in zip(images, masks):
        print('--> images from: ', i)
        print('-->  masks from: ', m)
        if 'c256' in i:
            dataset = 'celeba'
        elif 'inet256' in i:
            dataset = 'imagenet'
            if method in ['generative_inpainting', 'batfill', 'aot', 'medfe']:
                print(f'-->  SELECTED METHOD [{method}] DO NOT CONTAIN IMAGENET WEIGHTS!')
                continue
        elif 'p256' in i:
            dataset = 'places'            
        tar_name = os.path.join(out_dir_tar, os.path.basename(i))
        print('-->     outs to: ', tar_name)
        if os.path.isfile(tar_name):
            print('Tar output already exist! Not overwriting!')
            continue
        os.makedirs(image_folder)
        os.makedirs(mask_folder)
        os.system(f'tar -xf {i} -C {image_folder}')
        os.system(f'tar -xf {m} -C {mask_folder}') # masks do not have 'mask' in filename
        os.chdir(mask_folder)
        os.system('for f in *png ; do mv -- "$f" "mask_$f" ; done') # renaming masks
        os.chdir(pwd)
        os.system(f'mv {mask_folder}/*png {image_folder}') # moving masks to images
        tmp_dir = os.path.join(out_dir_tar, os.path.basename(i).split('.')[0])
        os.makedirs(tmp_dir, exist_ok=True)
        os.system(f'input_dir={image_folder} output_dir={tmp_dir} DATASET={dataset} sh run.sh') # testing
        os.chdir(tmp_dir)
        os.system(f'tar -cf {tar_name} *')
        os.chdir(pwd)

        os.system(f'rm -rf {image_folder} {mask_folder} {tmp_dir}')

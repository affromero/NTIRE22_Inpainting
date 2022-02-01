#!/bin/bash
# rm -rf $TMPDIR/inpainting_baselines
# cd $TMPDIR

# REPO=https://github.com/affromero/inpainting_baselines.git
# git clone $REPO
# cd inpainting_baselines/ICT

ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv --system-site-packages myenv
    source $ENV
    pip install -U pip
    pip install -r requirements.txt
    pip install gshell torch torchvision opencv-python
fi

# if [ -d "ckpts_ICT" ]; then
if [ -d "ckpts_ICT" ]; then
    echo "Pretrained models ckpts_ICT exist!"
else
    ln -s /cluster/home/roandres/Code/inpainting_baselines/ICT/ckpts_ICT . 
    # wget -O ckpts_ICT.zip https://www.dropbox.com/s/cqjgcj0serkbdxd/ckpts_ICT.zip?dl=1
    # unzip ckpts_ICT.zip
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

# DATASET='celeba'
echo "--> Testing $DATASET"
if [[ $DATASET == *"celeba"* ]]; then
    python main.py --input_image ${_input_dir} --save_place ${_output_dir} --sample_num 1 --FFHQ 
elif [[ $DATASET == *"places"* ]]; then
    python main.py --input_image ${_input_dir} --save_place ${_output_dir} --sample_num 1 --Places2_Nature 
elif [[ $DATASET == *"imagenet"* ]]; then
    python main.py --input_image ${_input_dir} --save_place ${_output_dir} --sample_num 1 --ImageNet     
fi
# images in png format
# --sample_num: how many outputs per image
# --ImageNet --FFHQ --Places2_Nature

rm -r ${output_dir}/AP
deactivate

# input_dir=~/Code/inpainting_baselines/ICT/celeba output_dir=~/Code/inpainting_baselines/ICT/celeba_output DATASET=imagenet sh run.sh
# input_dir=/cluster/project/cvl/gudiff/inpainting_baselines_test_images_celebA/andres_M5JvMdjHCs3mH4r5gAAL/inp output_dir=/cluster/project/cvl/gudiff/inpainting_baselines_test_images_celebA/andres_M5JvMdjHCs3mH4r5gAAL/srs sh run.sh
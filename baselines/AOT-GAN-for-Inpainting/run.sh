#!/bin/bash
ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv --system-site-packages myenv
    source $ENV
    pip install -U pip
    pip install gshell torch torchvision opencv-python
fi

# conda env create -f environment.yml 
# conda activate inpainting

if [[ $DATASET == *"celeba"* ]]; then
    pretrain=celebahq/G0000000.pt
    if [ -f "${pretrain}" ]; then
        echo "Pretrained models ${pretrain} exist!"
    else
        gshell init
        gshell download --with-id 1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ --recursive
    fi
elif [[ $DATASET == *"places"* ]]; then
    pretrain=places2/G0000000.pt
    if [ -f "${pretrain}" ]; then
        echo "Pretrained models ${pretrain} exist!"
    else
        gshell init
        gshell download --with-id 1bSOH-2nB3feFRyDEmiX81CEiWkghss3i --recursive
    fi
elif [[ $DATASET == *"imagenet"* ]]; then
    echo "No pretrained ImageNet!"
    exit 0
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

# DATASET='celeba'
echo "--> Testing $DATASET"
python src/run.py --dir_image ${_input_dir} --dir_output ${_output_dir}  --pre_train ${pretrain} --image_size 256
deactivate

# input_dir=~/Code/inpainting_baselines/AOT-GAN-for-Inpainting/celeba output_dir=~/Code/inpainting_baselines/AOT-GAN-for-Inpainting/celeba_output DATASET=celeba sh run.sh

#!/bin/bash
module load matlab/R2021a

ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv --system-site-packages myenv
    source $ENV
    pip install -U pip
    pip install gshell matlab torch torchvision opencv-python
fi

# conda env create -f environment.yml 
# conda activate inpainting

if [ -d "parameters" ]; then
    echo "Pretrained models exist!"
else
    gshell download --with-id 1uLC9YN_34mLod5kIE1nMb9P5L40Iqbkp --recursive
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

# DATASET='celeba'
echo "--> Testing $DATASET"
if [[ $DATASET == *"celeba"* ]]; then
    # celeba was only trained for rectangle crops, so free form fails
    # python test_folder.py --in_dir ${_input_dir} --out_dir ${_output_dir} --checkpoint_folder parameters/celeba
    python test_folder.py --in_dir ${_input_dir} --out_dir ${_output_dir} --checkpoint_folder parameters/place2 
elif [[ $DATASET == *"places"* ]]; then
    python test_folder.py --in_dir ${_input_dir} --out_dir ${_output_dir} --checkpoint_folder parameters/place2
elif [[ $DATASET == *"imagenet"* ]]; then
    echo "No ImageNet pretrained model!"
    exit 0
fi
deactivate

# input_dir=~/Code/inpainting_baselines/Rethinking-Inpainting-MEDFE/celeba output_dir=~/Code/inpainting_baselines/Rethinking-Inpainting-MEDFE/celeba_output DATASET=celeba sh run.sh

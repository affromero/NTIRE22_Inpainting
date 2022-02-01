#!/bin/bash
# rm -rf $TMPDIR/inpainting_baselines
# cd $TMPDIR

# REPO=https://github.com/affromero/inpainting_baselines.git
# git clone $REPO
# cd inpainting_baselines/Diverse-Structure-Inpainting

module load python/3.6.5
module load cuda/9.0.176 cudnn/7.1.4
ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv myenv
    source $ENV
    pip install pip -U
    pip install https://files.pythonhosted.org/packages/55/7e/bec4d62e9dc95e828922c6cec38acd9461af8abe749f7c9def25ec4b2fdb/tensorflow_gpu-1.12.0-cp36-cp36m-manylinux1_x86_64.whl # gpu
    # pip install https://files.pythonhosted.org/packages/22/cc/ca70b78087015d21c5f3f93694107f34ebccb3be9624385a911d4b52ecef/tensorflow-1.12.0-cp36-cp36m-manylinux1_x86_64.whl # cpu
    pip install gshell opencv-python Image
fi

# DATASET='celeba'
echo "--> Testing $DATASET"
if [[ $DATASET == *"celeba"* ]]; then
    model_logs=celebahq_random
    if [ -d "${model_logs}" ]; then
        echo "Pretrained models ${model_logs} exist!"
    else
        gshell init    
        gshell download --with-id 1jLGVwWREwfGaKEzsr8f4IUqCCFANkFvG --recursive
    fi
elif [[ $DATASET == *"places"* ]]; then
    model_logs=places2_random
    if [ -d "${model_logs}" ]; then
        echo "Pretrained models ${model_logs} exist!"
    else
        gshell init    
        gshell download --with-id 1h6tU-2P1j2DFAD42VntFS7XsKNRBI7__ --recursive
    fi
elif [[ $DATASET == *"imagenet"* ]]; then
    model_logs=imagenet_random
    if [ -d "${model_logs}" ]; then
        echo "Pretrained models ${model_logs} exist!"
    else
        gshell init    
        gshell download --with-id 1ZNh9vjZGevCjUg-mF08pT6L3KZLo8MTL --recursive
    fi
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

python test.py --test_num 1 --img_flist ${_input_dir} --checkpoints_dir ${_output_dir} --full_model_dir ${model_logs}
deactivate
# input_dir=~/Code/inpainting_baselines/Diverse-Structure-Inpainting/celeba output_dir=~/Code/inpainting_baselines/Diverse-Structure-Inpainting/celeba_output sh run.sh
#!/bin/bash
# rm -rf $TMPDIR/inpainting_baselines
# cd $TMPDIR

# REPO=https://github.com/affromero/inpainting_baselines.git
# git clone $REPO
# cd inpainting_baselines/Diverse-Structure-Inpainting

module load python/3.7.4
module load cuda/10.0.130 cudnn/7.6.4
ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv myenv
    source $ENV
    pip install pip -U
    pip install https://files.pythonhosted.org/packages/73/b5/adc281ce4e631251c749d342793795832026edf9035df81c3813ef33fad2/tensorflow_gpu-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl # gpu
    # pip install https://files.pythonhosted.org/packages/9a/51/99abd43185d94adaaaddf8f44a80c418a91977924a7bc39b8dacd0c495b0/tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl # cpu
    pip install gshell torch torchvision opencv-python
fi

DATASET='celeba'
echo "--> Testing $DATASET"
if [[ $DATASET == *"celeba"* ]]; then
    model_logs=co-mod-gan-ffhq-9-025000.pkl
    if [ -f "${model_logs}" ]; then
        echo "Pretrained models ${model_logs} exist!"
    else
        gshell init    
        gshell download --with-id 1b3XxfAmJ9k2vd73j-3nPMr_lvNMQOFGE --recursive
    fi
elif [[ $DATASET == *"places"* ]]; then
    model_logs=co-mod-gan-places2-050000.pkl
    if [ -f "${model_logs}" ]; then
        echo "Pretrained models ${model_logs} exist!"
    else
        gshell init    
        gshell download --with-id 1dJa3DRWIkx6Ebr8Sc0v1FdvWf6wkd010 --recursive
    fi
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

python run_test.py -d ${_input_dir} -c ${model_logs} -s ${_output_dir}
deactivate
# input_dir=~/Code/inpainting_baselines/co-mod-gan/celeba output_dir=~/Code/inpainting_baselines/co-mod-gan/celeba_output sh run.sh
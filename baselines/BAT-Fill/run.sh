#!/bin/bash
# rm -rf $TMPDIR/inpainting_baselines
# cd $TMPDIR

# REPO=https://github.com/affromero/inpainting_baselines.git
# git clone $REPO
# cd inpainting_baselines/BAT-Fill

ENV=myenv/bin/activate
export PYTHONPATH=
if [ -f "$ENV" ]; then
    source $ENV
else
    python -m venv --system-site-packages myenv
    source $ENV
    pip install -U pip
    pip install -r requirements.txt
    pip install gshell torch torchvision
fi

# DATASET='celeba'
echo "--> Testing $DATASET"
if [[ $DATASET == *"celeba"* ]]; then
    bat=celeba_bat_pretrain
    up=celeba_up_pretrain
    if [ -d "checkpoints/${bat}" ]; then
        echo "Pretrained models checkpoints/${bat} checkpoints/${up} exist!"
    else
        gshell init
        gshell download --with-id 1C0yQy4mUarxP5ym0aXyCf2Or_rbdWek0 --recursive
        gshell download --with-id 1W1f286VrKJF9E8hbE9B5ivoZ67_1ikVg --recursive    
        mkdir -p checkpoints/${bat} checkpoints/${up}
        mv latest_tran.pth checkpoints/${bat}
        mv latest_net_G.pth checkpoints/${up}
    fi
elif [[ $DATASET == *"places"* ]]; then
    bat=places_bat_pretrain
    up=places_up_pretrain
    if [ -d "checkpoints/${bat}" ]; then
        echo "Pretrained models checkpoints/${bat} checkpoints/${up} exist!"
    else
        gshell init
        gshell download --with-id 11LGiqd1rutYBN8FIz49hpukMuwN-MSeA --recursive
        gshell download --with-id 1xf4ZNOdB8WfuPFypg_wl1GnV__qAyKQr --recursive        
        mkdir -p checkpoints/${bat} checkpoints/${up}
        mv latest_tran.pth checkpoints/${bat}
        mv latest_net_G.pth checkpoints/${up}
    fi
elif [[ $DATASET == *"imagenet"* ]]; then
    echo "No pretrained ImageNet!"
    exit 0
fi

_output_dir=`realpath ${output_dir}`
_input_dir=`realpath ${input_dir}`

python bat_sample.py --num_sample 1 --tran_model ${bat} --up_model ${up} --input_dir ${_input_dir} --save_dir ${_output_dir}
deactivate

# input_dir=~/Code/inpainting_baselines/BAT-Fill/celeba output_dir=~/Code/inpainting_baselines/BAT-Fill/celeba_output sh run.sh
# python3 bin/predict.py model.path=$1 indir=$2 outdir=$2_LaMa_output

# conda activate /scratch_net/schusch/Andres/Code/Inpainting/baselines/AOT-GAN-for-Inpainting/envs
# export TORCH_HOME=$(pwd) && export PYTHONPATH=.

name='LaMa_pretrained'
# model_path="$(pwd)/experiments/FFHQ/roandres_2022-03-28_20-03-23_train_lama-fourier_"
root="/scratch_net/schusch/roandres/Code/Inpainting/datasets"

###############################################################
dataset='ImageNet'
model_path="$(pwd)/LaMa_models/big-lama"
indir="${root}/${dataset}/val"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

indir="${root}/${dataset}/test"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

###############################################################
dataset='FFHQ'
model_path="$(pwd)/LaMa_models/lama-celeba-hq/lama-fourier"
indir="${root}/${dataset}/val"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

indir="${root}/${dataset}/test"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

# ###############################################################
dataset='Places'
model_path="$(pwd)/LaMa_models/big-lama"
indir="${root}/${dataset}/val"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

indir="${root}/${dataset}/test"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

###############################################################
dataset='WikiArt'
model_path="$(pwd)/LaMa_models/big-lama"
indir="${root}/${dataset}/val"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

indir="${root}/${dataset}/test"
out_dir="${indir}_${name}"
python bin/predict.py model.path=$model_path indir=$indir outdir=$out_dir

# ###############################################################
# zip -r ${root}/val_${name}.zip ${root}/FFHQ/val_${name} ${root}/Places/val_${name} ${root}/ImageNet/val_${name} ${root}/WikiArt/val_${name}
# zip -r ${root}/test_${name}.zip ${root}/FFHQ/test_${name} ${root}/Places/test_${name} ${root}/ImageNet/test_${name} ${root}/WikiArt/test_${name}

# 'val': {
#     'FFHQ': '/scratch_net/schusch_second/FFHQ/ffhq-dataset/ffhq-resized-512x512_val',
#     'Places': '/scratch_net/schusch_second/Places/val_large',
#     'ImageNet': '/scratch_net/schusch_second/ImageNet/ILSVRC/Data/CLS-LOC/val',
#     'WikiArt': '/scratch_net/schusch_second/WikiArt/test',
# },
# 'test': {
#     'FFHQ': '/scratch_net/schusch_second/FFHQ/ffhq-dataset/ffhq-resized-512x512_val',
#     'Places': '/scratch_net/schusch_second/Places/test_large',
#     'ImageNet': '/scratch_net/schusch_second/ImageNet/ILSVRC/Data/CLS-LOC/test',
#     'WikiArt': '/scratch_net/schusch_second/WikiArt/test',
# }
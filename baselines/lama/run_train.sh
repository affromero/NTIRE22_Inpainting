conda activate /scratch_net/schusch/Andres/Code/Inpainting/baselines/AOT-GAN-for-Inpainting/envs
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
python bin/train.py -cn lama-fourier location=$1 data.batch_size=10

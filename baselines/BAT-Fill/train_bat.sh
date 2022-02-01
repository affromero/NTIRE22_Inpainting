python3 train_bat.py --name celebahq_bat --dataset_mode inpaint --dataroot ../../dataset/flist \
            --dataset_name celebahq2 --mask_type 4 --pconv_level 0 --load_size 32 --max_dataset_size 30000 \
            --netG Upsampler --batchSize 12 \
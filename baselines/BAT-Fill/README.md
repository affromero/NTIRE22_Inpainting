# Diverse Image Inpainting with Bidirectional and Autoregressive Transformers

## Installation
```bash
pip install -r requirements.txt
```

## Dataset Preparation
Given the dataset, please prepare the images paths in a folder named by the dataset with the following folder strcuture.
```
    flist/dataset_name
        ├── train.flist    # paths of training images
        ├── valid.flist    # paths of validation images
        └── test.flist     # paths of testing images
```
In this work, we use CelebA-HQ (Download availbale [here](https://github.com/switchablenorms/CelebAMask-HQ)), Places2 (Download availbale [here](http://places2.csail.mit.edu/download.html)), ParisStreet View (need author's permission to download)

**ImageNet K-means Cluster:** The `kmeans_centers.npy` is downloaded from [image-gpt](https://github.com/openai/image-gpt), it's used to quantitize the low-resolution images.

## Testing with Pre-trained Models

1. Download pre-trained models: 

- CelebA-HQ: [BAT](https://drive.google.com/file/d/1C0yQy4mUarxP5ym0aXyCf2Or_rbdWek0/view?usp=sharing) ; [Upsmapler](https://drive.google.com/file/d/1W1f286VrKJF9E8hbE9B5ivoZ67_1ikVg/view?usp=sharing)
- Places2: [BAT](https://drive.google.com/file/d/11LGiqd1rutYBN8FIz49hpukMuwN-MSeA/view?usp=sharing) ; [Upsmapler](https://drive.google.com/file/d/1xf4ZNOdB8WfuPFypg_wl1GnV__qAyKQr/view?usp=sharing)
- Paris-StreetView: [BAT](https://drive.google.com/file/d/1yhRT9TCkBznw6_K4nrED778qoR8DS4DZ/view?usp=sharing) ; [Upsmapler](https://drive.google.com/file/d/1gitMXnJN282S7PP3qOSqrscP_EoE8MWd/view?usp=sharing)
2. Put the pre-trained model under the checkpoints folder, e.g.
```
    checkpoints
        ├── celebahq_bat_pretrain
            ├── latest_net_G.pth 
```
3. Prepare the input images and masks to test.
```bash
python bat_sample.py --num_sample [1] --tran_model [bat name] --up_model [upsampler name] --input_dir [dir of input] --mask_dir [dir of mask] --save_dir [dir to save results]
```

## Training New Models
**Pretrained VGG model** Download from [here](https://drive.google.com/file/d/1fp7DAiXdf0Ay-jANb8f0RHYLTRyjNv4m/view?usp=sharing), move it to `models/`. This model is used to calculate training loss for the upsampler.

New models can be trained with the following commands.

1. Prepare dataset. Use `--dataroot` option to locate the directory of file lists, e.g. `./flist`, and specify the dataset name to train with `--dataset_name` option. Identify the types and mask ratio using `--mask_type` and `--pconv_level` options. 

2. Train the transformer. 
```bash
# To specify your own dataset or settings in the bash file.
bash train_bat.sh
```

Please note that some of the transformer settings are defined in `train_bat.py` instead of `options/`, and this script will take every available gpus for training, please define the GPUs via `CUDA_VISIBLE_DEVICES` instead of `--gpu_ids`, which is used for the upsampler.

3. Train the upsampler.
```bash
# To specify your own dataset or settings in the bash file.
bash train_up.sh
```
The upsampler is typically trained by the low-resolution ground truth, we find that using some samples from the trained BAT might be helpful to improve the performance i.e. PSNR, SSIM. But the sampling process is quite time consuming, training with ground truth also could yield reasonable results.


## Citation
If you find this code helpful for your research, please cite our papers.
```bash
@inproceedings{yu2021diverse,
  title={Diverse Image Inpainting with Bidirectional and Autoregressive Transformers},
  author={Yu, Yingchen and Zhan, Fangneng and Wu, Rongliang and Pan, Jianxiong and Cui, Kaiwen and Lu, Shijian and Ma, Feiying and Xie, Xuansong and Miao, Chunyan},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```


## Acknowledgments
This code borrows heavily from [SPADE](https://github.com/NVlabs/SPADE) and [minGPT](https://github.com/karpathy/minGPT), we apprecite the authors for sharing their codes. 
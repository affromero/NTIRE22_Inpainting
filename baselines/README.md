# Image inpainting Baselines

This repo is created to provide **EASY** comparison with **SOME** Image Inpainting methods. Feel free to create a pull request to add more. 

Also, keep in mind there are other methods that leverage on deep priors for the inpainting task such as [Deep Image Prior](https://arxiv.org/abs/1711.10925), [PsP](https://openaccess.thecvf.com/content/CVPR2021/html/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.html), and more novel Diffusion Models such as [SDEdit](https://arxiv.org/abs/2108.01073) or [RePaint](https://arxiv.org/abs/2201.09865).

### Run Baseline
Each folder has a `run.sh` that installs all the dependencies** and evaluate each method (**from their original implementations and their original pretrained weights**) according to three variables as follows:
```bash
input_dir=/path/to/inputdir output_dir=/path/to/outputdir DATASET=celeba sh run.sh
``` 
- `input_dir`: folder with both images and masks in a 1:1 ratio. Masks should have the name `mask` in the filename. It only supports `png` images.
- `output_dir`: it will contain the generated images with the same name as the input images.
- `DATASET`: depending on the selected method, it could be celeba, imagenet, or places.

**Modify `run.sh` to your needs.

### Test Tar files
Sometimes the test evaluation is given by several tar files with different masks and different images, as follows:
```
tar_folder
|_ images_gt_a.tar
|_ masks_gt_a.tar
|_ images_gt_b.tar
|_ masks_gt_b.tar
.
.
.
```
In such case, you should run `test.py` with the `method` as argument (eg., `python test.py --method ict`), and it will create a folder with the name of the method, as follows (for instance ICT method):
```
tar_folder
|_ images_ict256_gt_a.tar
|_ masks_ict256_gt_a.tar
|_ images_c256_gt_b.tar
|_ masks_c256_gt_b.tar
|_ ict
  |_ images_ict256_gt_a.tar
  |_ images_c256_gt_b.tar
.
.
.
```
It will dynamically change the dataset if the tar filename has `c256` for celeba and `inet256` for imagenet.
Modify `test.py` to your needs.

### Requirements
Each method requirement is installed internally. The only additional requirement is `gshell`, which is used to download the pretrained weights from Google Drive. It might ask you for your authorization in your google account.

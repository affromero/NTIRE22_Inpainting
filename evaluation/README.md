# Perceptual Evaluation
Given that Image Inpainting is an inverse problem, in which most solutions differ with the ground-truth at the pixel level, we use Perceptual Metrics to rank the participants. 

We use the Learned Perceptual Image Patch Similarity ([LPIPS](https://github.com/richzhang/PerceptualSimilarity)) and Frechet Inception Distance ([FID](https://github.com/GaParmar/clean-fid)) as perceptual metrics, as well as, the standard Peak Signal To Noise Ratio (PSNR) and the Structural Similarity (SSIM) index as often employed in the literature. PSNR and SSIM implementations are found in most of the image processing toolboxes. 

The validation server provides feedback in the form of PSNR and SSIM scores, the other measures will be computed offline and reported for the final ranking after the end of the challenge.

As a final ranking, we will select the champion based on the perceptual metrics as well as a Mean Opinion Score (MOS) for the top solutions. We will also report the runtime of each method, and it will not be included in the final ranking.

- In Track 1, the participants should inpaint the input image according to the input mask, and the evaluation is conducted between the inpainted image and the ground-truth image.
- In Track 2, the participants should inpaint the input image according to both the input mask and the input sketch, and the evaluation is conducted between the inpainted image and the ground-truth image and should be consistent with the semantic mask. Thus, a [semantic segmentation network](https://arxiv.org/abs/1706.05587) generates the semantic labels of the completed images, and we compute the mean Intersection over Union(mIoU) with reference to the ground truth semantic labels. 

For submitting the results, you need to follow these steps:

    save the inpainted images as PNG files in separate folders, with the same structure as the input, e.g., the inpainted images of Places/Completion/000009.png should be saved following the same structure ./Places/Completion/000009.png. On the online server we cannot use all the image results and all the datasets due to the submission file size constraints. Therefore, in the validation phase, we will invite the participants to submit their results with *ONLY* for the Places dataset, and in a uniform sampling each 10th frame. This means, images 000000.png, 000010.png, 000020.png, 000030.png, 000040.png, … for all the masks. In the test phase, all frames will be evaluated. An example submission for validation: https://data.vision.ee.ethz.ch/reyang/NTIRE2022/validation/val_eg_2.zip
    create a ZIP archive containing all the output image results named as above and a readme.txt.
    
    The readme.txt file should contain the following lines filled in with the runtime per image (in seconds) of the solution, 1 or 0 accordingly if employs CPU or GPU at runtime, and 1 or 0 if employs extra data for training the models or not.
    
        runtime per frame [s] : 11.23
        CPU[1] / GPU[0] : 1
        Extra Data [1] / No Extra Data [0] : 0
        Other description : Solution based on an efficient combination of LaMa [Suvorov et al., WACV 2021] and RePaint [Lugmayr et al., ArXiv 2022]. We have our own pytorch implementation, and report single core CPU runtime. The method was trained on the provided training sets using 4xV100 GPUs during two weeks.
    
    The last part of the file can have any description you want about the code producing the provided results (dependencies, link, scripts, etc.)
    
    The provided information is very important both during the validation period when different teams can compare their results / solutions but also for establishing the final ranking of the teams and their methods.



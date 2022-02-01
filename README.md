# Image Inpainting Challenge - Official Repository

## Important dates
- **2022.02.01**: Release of train data (input and output images) and validation data (only input)
- **2022.02.01**: Validation server online
- **2022.03.13**: Final test data release (only input images)
- **2022.03.20**: Test output results submission deadline
- **2022.03.20**: Fact sheets and code/executable submission deadline
- **2022.03.22**: Preliminary test results release to the participants
- **2022.04.01**: Paper submission deadline for entries from the challenge
- **2022.06.19**: Workshop day

## Description
<img src="https://data.vision.ee.ethz.ch/cvl/ntire22/assets/img/backgrounds/bg5.jpg" width=1000 height=300/>
The 7th edition of [NTIRE: New Trends in Image Restoration and Enhancement workshop will be held on June 2022 in conjunction with CVPR 2022](https://data.vision.ee.ethz.ch/cvl/ntire22).

Image manipulation is a key computer vision task, aiming at the restoration of degraded image content, the filling in of missing information, or the needed transformation and/or manipulation to achieve the desired target (with respect to perceptual quality, contents, or performance of apps working on such images). Recent years have witnessed an increased interest from the vision and graphics communities in these fundamental topics of research. Not only has there been a constantly growing flow of related papers, but also substantial progress has been achieved.
Image Inpainting, also known as image completion, is a key computer vision task that aims at filling missing information within an image. Recent years have witnessed an increased interest from the vision and graphics communities in this fundamental topic of research.

Recently, there has been a substantial increase in the number of published papers that directly or indirectly address Image Inpainting. Due to a lack of a standardized framework, it is difficult for a new method to perform a comprehensive and fair comparison with respect to existing solutions.
This workshop aims to provide an overview of the new trends and advances in those areas. Moreover, it will offer an opportunity for academic and industrial attendees to interact and explore collaborations.

Jointly with the NTIRE workshop, we have an NTIRE challenge on Image Inpainting, that is, the task of predicting the values of missing pixels in an image so that the completed result looks realistic and coherent. This challenge has 3 main objectives:

1. Direct comparison of recent state-of-the-art Image Inpainting solutions, which will be considered as baselines.
2. To perform a comprehensive analysis on the different types of masks, for instance, strokes, half completion, nearest neighbor upsampling, *etc*. Thus, highlighting the pros and cons of each method for each type of mask.
3. To set a public benchmark on 4 different datasets (Faces, Places, ImageNet, and WikiArt) for direct and easy comparison.

This challenge has 2 tracks:
- **Track 1**: Traditional Image Inpainting.
- **Track 2**: Image Inpainting conditioned on Semantic Segmentation mask.


## Main Goal
The aim is to obtain a **mask agnostic** network design/solution capable of producing high-quality results with the best perceptual quality with respect to the ground truth.

## Data
Following a common practice in Image Inpainting methods, we use three popular datasets for our challenge: [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Places](http://places2.csail.mit.edu), and [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php#). Additionally, to explore a new benchmark, we also use the [WikiArt](https://www.kaggle.com/c/painter-by-numbers/data) dataset to tackle inpainting towards art creation. See the [data](data/) for more info about downloading the datasets.

## Competition
The top-ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solutions and to submit to the associated NTIRE workshop at CVPR 2022.

## Evaluation
See [Evaluation](evaluation).

## Provided Resources
- Scripts: With the dataset, the organizers will provide scripts to facilitate the reproducibility of the images and performance evaluation results after the validation server is online. More information is provided on the data page.
- Contact: You can use the forum on the data description page ([Track1](https://codalab.lisn.upsaclay.fr/competitions/1607) and [Track 2](https://codalab.lisn.upsaclay.fr/competitions/1608) - highly recommended!) or directly contact the challenge organizers by email (me [at] afromero.co, a.castillo13 [at] uniandes.edu.co, and Radu.Timofte [at] vision.ee.ethz.ch) if you have doubts or any question.

## Issues and questions: 
In case of any questions about the challenge or the toolkit, feel free to open an issue on Github.

## Organizers
* [Andrés Romero](https://afromero.co) (roandres@ethz.ch)
* [Ángela Castillo](https://angelacast135.github.io/) (a.castillo13@uniandes.edu.co)
* [Radu Timofte](http://people.ee.ethz.ch/~timofter/) (radu.timofte@vision.ee.ethz.ch)

## Terms and conditions
The terms and conditions for participating in the challenge are provided [here](terms_and_conditions.md)

## Steps to infer:

0. Setup python environment. Requirements are:
```
torch=1.10.1
torchvision=0.11.2
cv2
tensorflow=1.15.0
PIL
tqdm
easydict
pandas
numpy
```

1. Download models:
```
python download_models.py
```

2. Do inference:
```
python infer.py -i <input_dir_with_masks> -o <output_images> -d FFHQ
```

and repeat this for other datasets ["Places", "ImageNet", "WikiArt"].

@Authors: Ritwik Das (ritwikdas54@gmail.com), Sanchit Hira (sanchithira76@gmail.com)
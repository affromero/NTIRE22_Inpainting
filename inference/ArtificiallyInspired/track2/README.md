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
python infer.py -i ./ntire22_inp_val -o ./ntire22_inp_val_output -d FFHQ
python infer.py -i ./ntire22_inp_test -o ./ntire22_inp_test_output -d FFHQ
```

and repeat this for other datasets ["Places"].

@Authors: Ritwik Das (ritwikdas54@gmail.com), Sanchit Hira (sanchithira76@gmail.com)
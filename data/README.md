# Download Datasets for Inpainting

## FFHQ 

- Repo: https://github.com/NVlabs/ffhq-dataset
- Kaggle command: `kaggle datasets download -d rahulbhalley/ffhq-1024x1024`

<img src="https://github.com/NVlabs/ffhq-dataset/blob/master/ffhq-teaser.png" width=500/>

## Places2

- Webpage: http://places2.csail.mit.edu
- Download Train: http://data.csail.mit.edu/places/places365/train_large_places365challenge.tar

<img src="http://places2.csail.mit.edu/imgs/places2_collage_black.jpg" width=500/>

## ImageNet

- Webpage: https://www.image-net.org/challenges/LSVRC/2012/index.php#
- Kaggle link: https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz
- Kaggle command: `kaggle competitions download -c imagenet-object-localization-challenge`

<img src="https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_1k.jpg" width=500/>

## WikiArt

- Kaggle: https://www.kaggle.com/c/painter-by-numbers/data
- Kaggle command: `kaggle competitions download -c painter-by-numbers` (It requires `pip install kaggle==1.5.3`)

<img src="https://miro.medium.com/max/1400/0*jJX7bymBZPNoN0qk" width=500/>

# Validation Set + Masks: https://polybox.ethz.ch/index.php/s/nBta4VE0uBjG65D

It can be access it with `wget` [7.2GB]. It does **not** contain Ground-Truth images. The full set will be release it after the challenge.

It contains 1,000 images for each type of mask (7 x 1,000), for each dataset.
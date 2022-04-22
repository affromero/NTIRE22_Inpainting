
# Mask Generation Demo


```python
import numpy as np
from utils import MaskGeneration, MergeMask
from PIL import Image
import random
image_dir = '../src'
input_file = f'{image_dir}/TribunaUffizi.jpg'
array = np.array(Image.open(input_file))
```

 <img src="../src/TribunaUffizi.jpg" width=500/>


```python
# Load all masks
mask_generation = MaskGeneration()
```

## Traditional Strokes


```python
mode = {
    'name': 'ThickStrokes',
    'size': 512,
}
np.random.seed(1)
gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 18.82%


<img src="../src/ThickStrokes.png" width=500/>


```python
mode = {
    'name': 'MediumStrokes',
    'size': 512,
}
np.random.seed(1)
gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 11.52%


<img src="../src/MediumStrokes.png" width=500/>


```python
mode = {
    'name': 'ThinStrokes',
    'size': 512,
}
np.random.seed(1)
gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 13.77%


<img src="../src/ThinStrokes.png" width=500/>

## Image Completion Masks


```python
mode = {
    'name': 'Every_N_Lines',
    'n': 2,
    'direction': 'horizontal'
}

gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 50.00%


<img src="../src/Every_N_Lines.png" width=500/>


```python
mode = {
    'name': 'Completion',
    'ratio': 0.5,
    'direction': 'horizontal',
    'reverse': True,
}

gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 50.04%


<img src="../src/Completion.png" width=500/>


```python
mode = {
    'name': 'Expand',
    'size': None, # None means half of input size
    'direction': 'interior' # interior is center crop inpainting, exterior is extrapolation
}

gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (1144, 1195, 3)
    Output Mask     : (1144, 1195)
    Inpainting Ratio: 76.07%


<img src="../src/Expand.png" width=500/>

## Super Resolution Inpainting


```python
mode = {
    'name': 'Nearest_Neighbor',
    'scale': 4,
}

gt, mask = mask_generation(array, mode, verbose=True)
# mask ->    255: inpaint   -    0: keep
out = MergeMask(gt, 255 - mask)
name = mode.get('name')
filename = f'{image_dir}/{name}.png'
Image.fromarray(out).save(filename)
```

    Input: (1144, 1195, 3)
    Output GT       : (4576, 4780, 3)
    Output Mask     : (4576, 4780)
    Inpainting Ratio: 93.75%


#### Check image size in previous cell!
<img src="../src/Nearest_Neighbor.png" width=500/>



# MGR-Road

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **MGR-Road**.

This code is based on [SAM-Road](https://github.com/htcr/sam_road). 

⚠️ Sorry, the Key Modifications in model_MGRroad.py will be released upon paper acceptance.

## Installation
You need the following:
- an Nvidia GPU with latest CUDA and driver.
- the latest pytorch.
- pytorch lightning.
- wandb.
- Go, just for the APLS metric.
- and pip install whatever is missing.

## Getting Started

### SAM Preparation
Download the ViT-B checkpoint from the official SAM directory. Put it under:  
```
-sam_road++  
--sam_ckpts  
---sam_vit_b_01ec64.pth  
```

### Data Preparation
Refer to the instructions in the sam_road repo to download City-scale and SpaceNet datasets.
Put them in the main directory, structure like:  
```
-sam_road++  
--cityscale  
---20cities  
--spacenet  
---RGB_1.0_meter  
```
And run python generate_labes.py under both dirs.

### Training
City-scale dataset:  

```
python train.py --config=config/toponet_vitb_512_cityscale.yaml  --seed 44
```

SpaceNet dataset:
```
python train.py --config=config/toponet_vitb_256_spacenet.yaml --seed 44
```

### Inference
```
python inferencer.py --config=path_to_the_same_config_for_training--checkpoint=path_to_ckpt  
```

### Test
For APLS and TOPO metrics, please refer to [Sat2Graph](https://github.com/songtaohe/Sat2Graph). 


## 📝 Key Modifications

The core logic is implemented in `model_MGRroad.py` (renamed to `model.py`), corresponding to the contributions in the paper:

*   **Region-Aware Sampling (`RegionSampler`):** Replaces point-wise sampling with rotation-invariant grid sampling to handle occlusions and curved roads.
*   **Soft Decoupling:** Implements Mask Guidance with Stop-Gradient in `SAMRoad.forward`.
*   **Collaborative Optimization:** Implements Layer-wise Learning Rate Decay (LLRD) in `configure_optimizers`.



## 🙏 Acknowledgements
We thank the authors of [SAM-Road](https://github.com/htcr/sam_road) for their open-source contribution.

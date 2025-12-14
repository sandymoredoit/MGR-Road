这是一个更加简短、直接的 `README.md`，重点说明如何基于 `sam_road` 原库使用你的代码。

---

# MGR-Road: Resolving Semantic-Geometric Conflict in End-to-End Road Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSL%202025-blue)](https://ieeexplore.ieee.org/)

This repository contains the official implementation of **MGR-Road** (IEEE GRSL, 2025).

This code is based on [SAM-Road](https://github.com/htcr/sam_road). We introduce **Collaborative Optimization**, **Soft Decoupling**, and **Region-Aware Sampling** to resolve the conflict between semantic segmentation and geometric topology reasoning.

## 🚀 Usage

Since this project modifies the model architecture of SAM-Road while keeping the data pipeline and training logic intact, you can set it up as a drop-in replacement.

### 1. Prerequisite
Clone the original SAM-Road repository and set up the environment:

```bash
git clone https://github.com/htcr/sam_road.git
cd sam_road
# Follow the installation and data preparation steps in the original README
```

### 2. Apply MGR-Road
Replace the original `sam_road/model.py` with the `model16.py` provided in this repository (rename it to `model.py`).

```bash
# Assuming you are in the root of sam_road
cp /path/to/MGR-Road/model16.py ./sam_road/model.py
```

### 3. Train & Evaluate
Run the standard training commands as described in SAM-Road:

```bash
# Example
python train.py --config configs/sam_road_cityscale.yaml
```

## 📝 Key Modifications

The core logic is implemented in `model16.py` (renamed to `model.py`), corresponding to the contributions in the paper:

*   **Region-Aware Sampling (`RegionSampler`):** Replaces point-wise sampling with rotation-invariant grid sampling to handle occlusions and curved roads.
*   **Soft Decoupling:** Implements Mask Guidance with Stop-Gradient in `SAMRoad.forward`.
*   **Collaborative Optimization:** Implements Layer-wise Learning Rate Decay (LLRD) in `configure_optimizers`.

## 📜 Citation

If you use this code, please cite our paper:

```bibtex
@article{mgrroad2025,
  title={MGR-Road: Resolving Semantic-Geometric Conflict in End-to-End Road Extraction},
  author={Zhang, Z. and others},
  journal={IEEE Geoscience and Remote Sensing Letters (GRSL)},
  year={2025},
  volume={XX},
  number={XX}
}
```

## 🙏 Acknowledgements
We thank the authors of [SAM-Road](https://github.com/htcr/sam_road) for their open-source contribution.

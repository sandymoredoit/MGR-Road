# MGR-Road

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper: **"MGR-Road: [Your Paper Title Here]"**.

This code is built upon [SAM-Road](https://github.com/htcr/sam_road). We introduce **Region-Aware Sampling**, **Soft Decoupling**, and **Collaborative Optimization** to achieve state-of-the-art performance in road extraction.

## 🚀 Features
- **Full Implementation**: Includes the complete training and inference pipeline.
- **Novel Modules**: Source code for `RegionSampler` and Soft Decoupling mechanisms.
- **Reproducibility**: Pre-configured scripts to reproduce the results in the paper.


## 📝 Key Components
The core contributions described in the paper are implemented in `model.py`:
*   **Region-Aware Sampling:** Handles occlusions and curved roads via rotation-invariant grid sampling.
*   **Soft Decoupling:** Mask Guidance with Stop-Gradient strategies.
*   **LLRD Optimization:** Layer-wise Learning Rate Decay in `configure_optimizers`.

## 🙏 Acknowledgements
We thank the authors of [SAM-Road](https://github.com/htcr/sam_road) for their open-source contribution.

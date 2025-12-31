# MGR-Road

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper: **"MGR-Road: Resolving Semantic-Geometric Conflict in End-to-End Road Extraction"**.

This code is built upon [SAM-Road](https://github.com/htcr/sam_road). 


## 📝 Key Components
The core contributions described in the paper are implemented in `model.py`:
*   **Region-Aware Sampling:** Handles occlusions and curved roads via rotation-invariant grid sampling.
*   **Soft Decoupling:** Mask Guidance with Stop-Gradient strategies.
*   **LLRD Optimization:** Layer-wise Learning Rate Decay in `configure_optimizers`.

## 🙏 Acknowledgements
We thank the authors of [SAM-Road](https://github.com/htcr/sam_road) for their open-source contribution.

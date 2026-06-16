# Deep Unsupervised Learning using Nonequilibrium Thermodynamics

This repository implements deep unsupervised learning techniques using nonequilibrium thermodynamics principles.

> **Note:** This is a **PyTorch reimplementation** of Sohl-Dickstein et al. (2015),
> *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* ([arXiv:1503.03585](https://arxiv.org/abs/1503.03585)) —
> the precursor to DDPM. The original paper code was written in Theano/Blocks/Fuel (Python 2),
> which are no longer maintained; this repo is a modern PyTorch rewrite.

## Environment Setup

Before preprocessing the dataset, ensure you have the required environment set up:

1. Create Conda Environment:  
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the Environment:  
   ```bash
   conda activate deep_thermo
   ```

Detailed Korean developer documentation is available in [`docs/README.md`](docs/README.md).

---
# MNIST
---

## Train

### Prepare the Dataset

No manual preparation is required: `scripts/train.py` calls
`torchvision.datasets.MNIST(..., download=True)`, so MNIST is downloaded
automatically into `data/MNIST/` on first run.


### Run the Training Script

Execute the training script to start the training process:

1. **Give execution permission to the training script:**  
   ```bash
   chmod +x train.sh
   ```
2. **Run the training script:**  
   ```bash
   ./train.sh
   ```

## Sampling / Inference

> **Note:** `scripts/infer.py` currently provides **utility functions only**
> (`diffusion_step`, `generate_inpaint_mask`, `plot_images`) and has **no `__main__`
> entry point** — running `python scripts/infer.py` does nothing on its own.

Samples are generated automatically **during training** and logged to TensorBoard
(the `samples/generated` panel); they are produced by `DiffusionModel.sample()`
in `networks/dpm.py`.

To generate samples manually, load a trained checkpoint and call
`model.sample(batch_size)` (`networks/dpm.py`) or `generate_samples(model, n_samples)`
(`utils.py`). The inpainting/denoising helpers in `scripts/infer.py` can drive a
guided reverse process. Note the sampler currently stops at `min_t` (default 100)
rather than `t = 0` — see the project wiki for details.

---
Swiss Roll
---

### Run the Training Script and Infering Script

Execute the training script to start the training process:

1. **Give execution permission to the training script:**  
   ```bash
   chmod +x run_swiss_roll.sh
   ```
2. **Run the training script:**  
   ```bash
   ./run_swiss_roll.sh
   ```

## References

- The dataset is based on the MNIST and Swiss Roll dataset, which is widely used for training and testing in machine learning.
- Deep Unsupervised Learning using Nonequilibrium Thermodynamics (arXiv 2015)
   - [Review](https://outta.tistory.com/109) <br>
   - [Paper](https://arxiv.org/abs/1503.03585) <br>
   - [Code](https://github.com/youngunghan/2025-OUTTA-Gen-AI/blob/main/Reviews/Diffusion/Deep_Unsupervised_Learning_using_Nonequilibrium_Thermodynamics_mschoi.ipynb)

## About

This repository presents a novel approach to unsupervised learning using nonequilibrium thermodynamics, achieving efficient and effective learning from data.

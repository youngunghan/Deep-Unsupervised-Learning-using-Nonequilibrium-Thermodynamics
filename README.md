# Deep Unsupervised Learning using Nonequilibrium Thermodynamics

This repository implements deep unsupervised learning techniques using nonequilibrium thermodynamics principles. It includes scripts for dataset preprocessing, training, evaluation, and inference.

## Dataset Preprocessing

### Environment Setup

Before preprocessing the dataset, ensure you have the required environment set up:

1. Create Conda Environment:  
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the Environment:  
   ```bash
   conda activate deep_thermo
   ```

### Dataset Structure

The dataset should be organized in the following structure:

data/MNIST/raw/
├── train/
│ ├── 000001.png
│ ├── 000002.png
│ └── ...
└── test/
├── 000001.png
├── 000002.png
└── ...


## Train

### Prepare the Dataset

Ensure the MNIST dataset is preprocessed and available in the `data/MNIST/` directory.


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

## Inference

### Run the Inference Script

Execute the inference script to generate predictions on the MNIST dataset:

1. **Give execution permission to the inference script:**  
   ```bash
   chmod +x scripts/infer.py
   ```
2. **Run the inference script:**  
   ```bash
   python scripts/infer.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- The dataset is based on the MNIST dataset, which is widely used for training and testing in machine learning.

## About

This repository presents a novel approach to unsupervised learning using nonequilibrium thermodynamics, achieving efficient and effective learning from data.

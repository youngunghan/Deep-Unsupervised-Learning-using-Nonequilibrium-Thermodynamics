# Deep Unsupervised Learning using Nonequilibrium Thermodynamics

This repository implements deep unsupervised learning techniques using nonequilibrium thermodynamics principles. 

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

---
# MNIST
---

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

### Run the Training Script

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- The dataset is based on the MNIST dataset, which is widely used for training and testing in machine learning.

## About

This repository presents a novel approach to unsupervised learning using nonequilibrium thermodynamics, achieving efficient and effective learning from data.

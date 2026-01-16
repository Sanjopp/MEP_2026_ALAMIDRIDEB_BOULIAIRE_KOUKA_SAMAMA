# How Does Batch Normalization Really Help Optimization?

**Advanced Machine Learning Course Project**

This project revisits the fundamental question: **How does Batch Normalization really help optimization?** We conduct controlled experiments in a simplified setting using two architectures—Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN)—trained with and without Batch Normalization on CIFAR-10 and Fashion-MNIST datasets. Our experiments aim to isolate and verify the core hypothesis that **Batch Normalization facilitates optimization primarily through landscape smoothing** rather than by stabilizing activation statistics.

## 📋 Repository Overview

This project systematically investigates the impact of **Batch Normalization** through controlled experiments:

**Experimental Matrix**: 2 Datasets × 2 Architectures × 2 Analysis Levels = 8 Core Experiments
- **Datasets**: CIFAR-10 (RGB, 32×32) and Fashion-MNIST (Grayscale, 28×28)
- **Architectures**: MLP (fully-connected layers) and CNN (convolutional layers)
- **Analysis Granularity**: 
  - **Epoch-level** - Metrics computed after each complete epoch
  - **Step-level** - Metrics computed at every batch during training for fine-grained dynamics

Each experiment compares training **with** and **without** Batch Normalization, examining:
- Convergence speed
- Accuracy
- Loss stability during training
- Gradient flow and optimization dynamics
- Activation distribution changes

## 🗂️ Project Structure

### Core Files
- **`models/`** - Neural network architectures
  - `mlp.py` - Fully-connected (Dense) networks with optional BatchNorm
  - `cnn.py` - Convolutional Neural Networks with optional BatchNorm
  - `batchnorm.py` - Custom BatchNorm1D implementation from scratch
  
- **`train_utils.py`** - Training utilities and step-level logging
  - Gradient norm computation
  - Step-by-step training with per-batch metrics
  - Training and validation loops
  
- **`data.py`** - CIFAR-10 dataset utilities
- **`data_fashion.py`** - Fashion-MNIST dataset utilities
- **`check_data.ipynb`** - Data exploration and validation notebook

### Data
- **`DATA/cifar-10-batches-py/`** - CIFAR-10 dataset (binary format)
- **`DATA/FashionMNIST/`** - Fashion-MNIST dataset

### Experiments (Notebooks)

All experiments are organized in the `notebook/` directory with a systematic structure:

**Experiment Design Matrix:**
- 2 Datasets: CIFAR-10, Fashion-MNIST
- 2 Architectures: MLP (fully-connected), CNN (convolutional)
- 2 Analysis Levels: Epoch-level, Step-level (per-batch granularity)

#### Fashion-MNIST + MLP
1. **`experiment_fashion.ipynb`** - Epoch-level analysis
   - Training MLP on Fashion-MNIST
   - Compares with/without Batch Normalization
   - Metrics per epoch: Loss, Accuracy

2. **`experiment_fashion_by_steps.ipynb`** - Step-level analysis
   - Same MLP but tracks metrics at every batch (step)
   - Per-step loss, accuracy, and gradient norms
   - Reveals fine-grained training dynamics

#### Fashion-MNIST + CNN
3. **`experiment_fashion_cnn.ipynb`** - Epoch-level analysis
   - Convolutional approach on Fashion-MNIST
   - Feature extraction benefits and MLP comparison
   - Epoch-level performance comparison

4. **`experiment_fashion_cnn_by_steps.ipynb`** - Step-level analysis
   - Same CNN architecture with step-by-step logging
   - Detailed gradient flow and loss evolution per batch

#### CIFAR-10 + MLP
5. **`experiment_cifar.ipynb`** - Epoch-level analysis
   - Training MLP on CIFAR-10 dataset
   - with/without Batch Normalization comparison
   - Standard learning curves per epoch

6. **`experiment_cifar_by_steps.ipynb`** - Step-level analysis
   - Step-by-step optimization dynamics on CIFAR-10
   - Per-batch metrics for detailed analysis

#### CIFAR-10 + CNN
7. **`experiment_cifar_cnn.ipynb`** - Epoch-level analysis
   - CNN models on CIFAR-10
   - Higher complexity dataset with convolutional features
   - Epoch-level performance metrics

8. **`experiment_cifar_cnn_by_steps.ipynb`** - Step-level analysis
   - Fine-grained step analysis for CNN on CIFAR-10
   - Shows layer-wise evolution during training

#### Special Analysis
- **`experiment_activation_distribution.ipynb`**
  - Deep dive into Batch Normalization's effect on activation distributions
  - Visualizes how internal representations change
  - Demonstrates normalization across all tested configurations

## 🧠 Key Components

### Custom Batch Normalization
The `batchnorm.py` module implements a custom BatchNorm1D layer using PyTorch's autograd:
- Computes normalization statistics (mean, variance)
- Normalizes activations
- Includes learnable scale (γ) and shift (β) parameters
- Proper gradient computation for all parameters

### Models

**MLP (Multi-Layer Perceptron)**
```python
MLP(
    input_dim=784,        # For 28×28 images (Fashion-MNIST)
    hidden_dims=(512, 256, 128),
    num_classes=10,
    use_bn=True/False     # Toggle Batch Normalization
)
```

**ConvMLP (Convolutional + Dense)**
```python
ConvMLP(
    in_channels=3,        # 3 for CIFAR-10, 1 for Fashion-MNIST
    input_size=32,        # 32 for CIFAR-10, 28 for Fashion-MNIST
    conv_channels=(32, 64),
    hidden_dims=(512, 256, 128),
    num_classes=10,
    use_bn=True/False
)
```

### Training Utilities
- **`train_steps()`** - Trains for one epoch with per-batch logging
- **`grad_norm()`** - Computes L2 norm of all gradients
- **Per-step metrics** - Loss, accuracy, gradient norm tracked at batch level

## 📊 Datasets

### CIFAR-10
- 32×32 RGB images
- 10 object classes
- 50,000 training samples, 10,000 test samples

### Fashion-MNIST
- 28×28 grayscale images
- 10 clothing/object classes
- 60,000 training samples, 10,000 test samples

## 🔧 Requirements

All dependencies are listed in `requirements.txt`:
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **Jupyter** - Interactive notebooks
- **tqdm** - Progress bars
- **PIL** - Image processing
- CUDA support for GPU acceleration


## 📈 Key Findings (Expected)

Through these experiments, you'll observe:

1. **Faster Convergence** - Models with BatchNorm typically converge faster
2. **Higher Accuracy** - BN often leads to better final performance
3. **Smoother Training** - Loss curves are more stable with BatchNorm
4. **Better Gradient Flow** - Gradient norms remain more stable

## 🔬 Research Questions Addressed

- How does Batch Normalization affect activation distributions?
- Does BN improve training dynamics at the step level?
- Are the benefits consistent across different architectures (MLP vs CNN)?
- Does BN improve performance on different datasets?
- How do gradients flow differently with vs. without BN?

## 📝 Notes

- All experiments use standard SGD/Adam optimizers
- Batch size: typically 128
- Models are trained from scratch (no pre-training)

## 🔗 References

Related concepts:
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167) - Original BN introduction
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) - Analysis of BN's role in landscape smoothing vs. activation stabilization

---

**Project Status**: Active research and experimentation  
**Current Branch**: `main` - CNN implementations and analysis  
**Last Updated**: January 2025

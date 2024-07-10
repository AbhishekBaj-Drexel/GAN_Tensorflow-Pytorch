# Generative Adversarial Network (GAN) for MNIST Digit Generation

This project demonstrates the implementation and training of Generative Adversarial Networks (GANs) using both PyTorch and TensorFlow to generate handwritten digits, specifically focusing on the digit '0' from the MNIST dataset.

## Overview

Generative Adversarial Networks (GANs), introduced by Goodfellow et al. in 2014, consist of two neural networks, the Generator and the Discriminator, that are trained simultaneously through an adversarial process. The Generator aims to produce realistic data samples, while the Discriminator evaluates these samples to distinguish between real and fake data.

This project includes implementations of GANs using both PyTorch and TensorFlow to generate the digit '0' from the MNIST dataset. It also includes custom implementations for activation functions, batch normalization, and the training loop.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- NumPy
- TensorFlow
- PyTorch
- Matplotlib

You can install the required libraries using pip:

```sh
pip install numpy tensorflow torch matplotlib

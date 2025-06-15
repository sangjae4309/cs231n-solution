<h1 align="center">CS231n: Assignment Solutions</h1>
<p align="center"><i>Stanford - Spring 2025</i></p>

## About
### Overview
Complete solution code of standford cs231n assignemnt (spring 2025).
Check out a detailed walkthrough at **[this link](https://sangjae4309.github.io/projects/cs231n.html)**. The post contains more comprehensive explanations.

### Main sources 
* [**(official) Course page**](http://cs231n.stanford.edu/index.html)
* [**(official) Assignments1**](https://cs231n.github.io/assignments2025/assignment1/)
* [**(official) Assignments2**](https://cs231n.github.io/assignments2025/assignment1/)
* [**(official) Assignments3**](https://cs231n.github.io/assignments2025/assignment1/)
* [**Sangjae's solution blog post**](https://sangjae4309.github.io/projects/cs231n.html)

<br>

## Solutions
### Assignment 1
* [Q1](assignment1/knn.ipynb): k-Nearest Neighbor classifier. (_Done_)
* [Q2](assignment1/softmax.ipynb): Implement a Softmax classifier. (_Done_)
* [Q3](assignment1/two_layer_net.ipynb): Two-Layer Neural Network. (_Done_)
* [Q4](assignment1/features.ipynb): Higher Level Representations: Image Features. (_Done_)
* [Q5](assignment1/FullyConnectedNets.ipynb): Fully-connected Neural Network. (_Done_)

### Assignment 2
* [Q1](assignment2/BatchNormalization.ipynb): Batch Normalization. (_Done_)
* [Q2](assignment2/Dropout.ipynb): Dropout. (_Done_)
* [Q3](assignment2/ConvolutionalNetworks.ipynb): Convolutional Networks. (_Done_)
* [Q4](assignment2/PyTorch.ipynb) : PyTorch on CIFAR-10. (_Done_)
* [Q5](assignment2/RNN_Captioning_pytorch.ipynb): Image Captioning with Vanilla RNNs (_Done_)

### Assignment 3
* [Q1](assignment3/Transformer_Captioning.ipynb): Image Captioning with Transformers (_Done_)
* [Q2](assignment3/Self_Supervised_Learning.ipynb): Self-Supervised Learning for Image Classification (_Done_)
* [Q3](assignment3/DDPM.ipynb): Denoising Diffusion Probabilistic Models (In progress)
* [Q4](assignment3/CLIP_DINO.ipynb): CLIP and Dino (In progress)

<br>

## Running Locally

Instead of relying on Google Colab, I’ve set things up to run on a local GPU environment for better control and performance. All necessary dependencies and environment configurations are predefined in a Dockerfile.
<br>
After chaning directory to `cs231n-solution`. build the docker
```bash
  docker build --tag pytorch-gpu .
  docker run --gpus=all -d -it --privileged  --name pytorch-gpu pytorch-gpu
```

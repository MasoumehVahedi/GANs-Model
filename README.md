# Generative Adversarial Network
This project comes from Kaggle Competitons called Generative-Dog-Images and Anime Face Dataset. 
There have implemented three types of GANs named DCGAN, WGAN-GP, and CGAN. The architectures of these are briefly introduced in the following.

## 1- Deep Convolutional Generative Adversarial Network architecture (DCGAN)

In this project, we implemented DCGAN architecture using PyTorch deep learning framework.

### DCGAN Model Architecture

In this section, we go through, firstly, how we should build the discriminator and generator networks. Secondly, what are the parameters and hyperparameters.

**max-pooling layer**
DCGAN architecture is a fully convolutional network meaning that max-pooling will not be used for downsampling. All of the operations will be through strided convolutions only.

**Batch Normalization**
To normalize input to each unit of a layer, we will use Batch Normalization for both the discriminator and the generator.

**Activation function**
For Generator model, we will use the ReLU activation function in all the layers, except for the last one. For the last convolutional layer, we will use Tanh activation function.
While for Discriminator model, we will use LeakyReLU for all the convolutional layer after applying batch normalization.
In general, according to original paper, the main features of DCGAN:
1- Use convolutions without any pooling layers\
2- Use batchnorm in both the generator and the discriminator\
3- Don't use fully connected hidden layers\
4- Use ReLU activation in the generator for all layers except for the output, which uses a Tanh activation.\
5- Use LeakyReLU activation in the discriminator for all layers except for the output, which does not use an activation

## 2- Wasserstein GAN with Gradient Penalty (WGAN-GP)

In this section, we are going to implement a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some issues related to stability in GANs model.
BCE Loss is used traditionally to train GANs. However, it has many problems due the form of the function it's approximated by.

There are some advantages and disadvantages of WGAN-GP:

**Advantages:**

1 - This type of GAN has better training stability\
2- Loss function: a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse. 

**Disadvantages:**
Longer to train

### W-Loss vs BCE Loss

**W-Loss** : Critic output any number\
W-Loss helps with mode collapse and vanishing gradient problems

**BCE Loss** : Discriminator outputs between 0 and 1

**Note:** 
Some of the main differences between BCELoss and W-Loss is that, the discriminator under the BCE Loss outputs a value between 0 and 1, while the critic in W-Loss will output any number.

## 3-Conditional GAN (CGAN):

The conditional generative adversarial network, or cGAN, is a type of GAN that involves the conditional generation of images by a generator model.
Image generation can be conditional on a class label, allowing the targeted generated of images of a given type.
Two main reasons of making CGAN model to use the class label information in GANs model:
    
1. Improve the GAN:
Class labels, an additional information correlated with input images, are be able to use for improving the GAN model.
In this way, improving such as faster training, stable training, and better quality of generated images.
 2. Targeted Image Generation:
 we can use class labels for targeted generated images of a given type.

# Datasets
## 1- Anime Face Dataset

The Anim Face dataset includes  63,632 "high-quality" images which is used for generating anime faces using GAN model.

**Goal**

In this project, using definition and training a DCGAN, we are going to generate Anime fake faces as real images.
The main objective here is that make a generator network to generate new images that look like as real.

## 2- Dog dataset
Generative Dog Images is based on the Stanford Dogs Dataset. The folder includes:\
1- all-dogs.zip - All dog images contained in the Stanford Dogs Dataset\ 
2- Annotations.zip - Class labels, Bounding boxes

# Reference
•	Generative Dog Images Dataset: https://www.kaggle.com/c/generative-dog-images/data

•	Anime Face Dataset: https://www.kaggle.com/splcher/animefacedataset

•	Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016): https://arxiv.org/abs/1511.06434

•	Wasserstein GAN (Arjovsky, et al., 2017): https://arxiv.org/abs/1701.07875

•	Conditional Generative Adversarial Nets (Mirza and Osindero., 2014): https://arxiv.org/abs/1411.1784



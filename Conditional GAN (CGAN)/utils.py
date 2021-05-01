######### Class Input #########

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

"""
   There are some points to build Conditional GANs model:
   
   1- To create a Conditional GAN, we need to have the class information using a one-hot encoded vector
      as the input vector to build the Generator model. In fact, the length of one-hot encoded vector represents
      the number of classes and each index is the related class.
      Therefore, here, we should create one-hot vectors for each label.
         
           
    2- After one_hot encoding, before giving it to generator model, we should concatenate noise vector and
       one_hot class vector.
       Moreover, we should concatenate noise and one_hot vector when we want to add CLASS CHANNELS to the
       Discriminator model.
       Thus, we need to create a function to combine noise vector and one_hot vector.
          
"""

# One-hot class vector
def one_hot_labels(labels, num_classes):
    """
    One-hot-labels function is to create one-hot vectors for the labels
      which returns a tensor of shape (?, num_classes).
      Parameters:
           labels: tensor of labels from the dataloader, size (?)
           n_classes: the total number of classes in the dataset, an integer scalar

           Note: we need to convert labels to integer, otherwise we will get this ERROR:
                 one_hot is only applicable to index tensor
    """
    oh_encoding = F.one_hot(labels.to(torch.int64), num_classes)
    return oh_encoding


# Combining vectors (noise vector and one-hot vector)
def vectors_embedding(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
           Parameters:
                 x: (n_samples, ?) the first vector.
                     In this assignment, this will be the noise vector of shape (n_samples, z_dim),
                     but you shouldn't need to know the second dimension's size.
                 y: (n_samples, ?) the second vector.
                     Once again, in this assignment this will be the one-hot class vector
                     with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    # Note: The output of this function must be a float, so we change the format float
    embedding = torch.cat((x.float(), y.float()), 1)
    return embedding

"""
   Note: For Conditional GANs, we have to calculate the size of input vector in the Generator model. 
   In other words,the input of generator model is the noise vector which need to be concatenated with
   the class vector.
   One the other hand, we need to add a channel for each class in the Discriminator model
   The input_dimension Function will require the concept above.
"""
def input_dimension(z_dim, img_shape, num_classes):
    """
    This function is to get the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        img_shape: the shape of each DOG image as (C, W, H), which is (3, 64, 64)
        num_classes: the total number of classes in the dataset, an integer scalar
                (120 for this dataset)
    Returns:
        input_dim_generator: the input dimension of the conditional generator,
                             taking the noise and class vectors
        img_channel_discriminator: the number of input channels to the discriminator
                            (e.g. C x 64 x 64 for DOG images)
    """
    input_dim_generator = z_dim + num_classes
    img_channel_discriminator = img_shape[0] + num_classes
    return input_dim_generator, img_channel_discriminator


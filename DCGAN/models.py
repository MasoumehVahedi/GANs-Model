
import torch
import torch.nn as nn

""" Generator

      The first step is to create Generator model.

      In this case, it should be noticed that instead of passing in the image dimension, 
      we will pass the number of image channels to the generator. 
      This is because with DCGAN, we use convolutions which don’t depend on the number of pixels on an image. 
      However, the number of channels is important to determine the size of the filters.

      We will build a generator using 4 layers (3 hidden layers + 1 output layer). 
      As before, we will need to write a function to create a single block for the generator's neural network. 
      Since in DCGAN the activation function will be different for the output layer, 
      we will need to check what layer is being created. We are supplied with some tests following the code cell 
      so you can see if you're on the right track!

      At the end of the generator class, we are given a forward pass function that takes in a noise vector and 
      generates an image of the output dimension using the neural network.
"""


# Build Generator class
class Generator(nn.Module):
    """ Generator Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (Anim_Face is rgb, so 3 is our default)
    hidden_dim: the inner dimension, a scalar """

    def __init__(self, z_dim, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            # input: z_dim x 1 x 1
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),  # out: 512 x 4 x 4
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # out: 256 x 8 x 8
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            # out: 128 x 16 x 16
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),  # out: 64 x 32 x 32
            # Final layer: we do not use BatchNorm
            # Also, the activation function in last layer will be Tanh
            # The second parameter should be the number of image channel
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1),  # out: 3 x 64 x 64
            nn.Tanh()  # Normalize images between -1 and +1
        )

    # Build the neural block
    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            # In Generator, we use Relu finction
            nn.ReLU(inplace=True)
        )

    def unsqeeze_noise(self, noise):
        '''
           Function for completing a forward pass of the generator: Given a noise tensor,
           returns a copy of that noise with width and height = 1 and channels = z_dim.
           Parameters:
              noise: a noise tensor with dimensions (n_samples, z_dim)
              view : a function for reshaping
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
           Function for completing a forward pass of the generator: Given a noise tensor,
           returns generated images.
           Parameters:
              noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqeeze_noise(noise)
        return self.gen(x)


""" Discriminator

     The second step that we should construct is the discriminator. As with the generator model, 
     we will begin by creating a function that builds a neural network block for the discriminator.

     Note: You use leaky ReLUs to prevent the "dying ReLU" problem, which refers to the phenomenon 
     where the parameters stop changing due to consistently negative values passed to a ReLU, 
     which result in a zero gradient.

"""

# Build Discriminator class
class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input shape: 3 x 64 x 64
            # According to original paper, we do not use BatchNorm layer in the first layer
            nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1
                      ),  # out: 64 x 32 x 32
            nn.LeakyReLU(0.2),
            # Middile layers
            self.make_disc_block(hidden_dim, hidden_dim * 2),  # out: 128 x 16 x 16
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4),  # out: 256 x 8 x 8
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8),  # out: 512 x 4 x 4
            # Final Layer
            # The second parameter should be single channel (1) because we want to represent one value which
            # the images is fake or real
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=2, padding=0)  # out: 1 x 1 x 1
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,  # Note: out_channels=3, because Anime_Face is RGB image
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        """ Function for completing a forward pass of the discriminator: Given an image tensor,
            returns a 1-dimension tensor representing fake/real.
            Parameters:
                image: a flattened image tensor with dimension (im_dim) """
        disc_pred = self.disc(image)
        images = disc_pred.view(len(disc_pred), -1)
        return images

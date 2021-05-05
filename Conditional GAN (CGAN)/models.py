# CGAN MODEL

import torch
import torch.nn as nn

# Genetator and noise
# The first step is to create Generator model.
class Generator(nn.Module):
    """
    Generator class values:
      z_dim: the dimension of the noise vector, a scaler
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (Anim_Face is rgb, so 3 is our default)
      hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, input_dim, img_channels=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),               # out: 512 x 4 x 4
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),          # out: 256 x 8 x 8
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),          # out: 128 x 16 x 16
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),              # out: 64 x 32 x 32
            # Final layer
            # The second parameter should be the number of image channel
            nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    # Build neural block
    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding):
        """ Function to return a sequence of operations corresponding to a generator block of CGAN.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
            """
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


# Discriminator
# The second step is to construct the discriminator.
class Discriminator(nn.Module):
    """
      Discriminator Class
      Values:
        im_chan: the number of channels of the output image, a scalar
              (ANIME_FACE is RGB, so 3 channel is the default)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, img_channels, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input shape: N x img_channels x 64 x 64
            # First layer : no BatchNorm layer
            nn.Conv2d(img_channels, hidden_dim, kernel_size=4, stride=2, padding=1),                     # out: 64 x 32 x 32
            nn.LeakyReLU(0.2),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),        # out: 128 x 16 x 16
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),    # out: 256 x 8 x 8
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),    # out: 512 x 4 x 4v
            # Final layer
            # The second parameter should be single channel (1) because we want to represent one value which
            # the images is fake or real
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=2, padding=0)                  # out: 1 x 1 x 1
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, image):
        """
           Forward function is for completing a forward pass of the discriminator: it is given an image tensor,
           then returns a 1-dimension tensor representing fake/real.
           Parameters:
               image: a flattened image tensor with dimension (im_chan)
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
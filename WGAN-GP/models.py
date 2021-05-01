
import torch
import torch.nn as nn

""" 
    Genetator and noise:
    The first step is to create Generator model. 
    The structure of generator and discriminator is similar to DCGAN, but in adjusting hyperparameters (i.e. Loss function) 
    we have some difference to get the desired result.
"""

class Generator(nn.Module):
    '''
       Generator class values:
         z_dim: the dimension of the noise vector, a scaler
         im_chan: the number of channels in the images, fitted for the dataset used, a scalar
                (Anim_Face is rgb, so 3 is our default)
         hidden_dim: the inner dimension, a scalar
         '''
    def __init__(self, z_dim, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            # input: z_dim x 1 x 1
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),                  # out: 512 x 4 x 4
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),         # out: 256 x 8 x 8
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),         # out: 128 x 16 x 16
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),             # out: 64 x 32 x 32
            # Final layer
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    # Build the neural block
    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding):
        '''
            Function to return a sequence of operations corresponding to a generator block of DCGAN;
            a transposed convolution, a batchnorm (except in the final layer), and an activation.
            Parameters:
               input_channels: how many channels the input feature representation has
               output_channels: how many channels the output feature representation should have
               kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
               stride: the stride of the convolution
               '''
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise):
        '''
            Forward function is related to complete a forward pass of the generator: This is given a noise tensor,
            hen returns generated images.
            Parameters:
                noise: a noise tensor with dimensions (n_samples, z_dim)
                '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)



"""   
    Discriminator (or Critic):
    The second step is to construct the discriminator.
"""

class Critic(nn.Module):
    ''' Critic Class
        Values:
               im_chan: the number of channels of the output image, a scalar
                  (ANIME_FACE is RGB, so 3 channel is the default)
               hidden_dim: the inner dimension, a scalar
               '''
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Critic, self).__init__()
        # Build the neural network
        self.crit = nn.Sequential(
            # Input shape: 3 x 64 x 64
            # First layer: no BatchNorm layer
            nn.Conv2d(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),                                                                           # out: 64 x 32 x 32
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),        # out: 128 x 16 x 16
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4,  kernel_size=4, stride=2, padding=1),   # out: 256 x 8 x 8
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8,  kernel_size=4, stride=2, padding=1),   # out: 512 x 4 x 4
            # Final layer
            nn.Conv2d(hidden_dim * 8, im_chan, kernel_size=4, stride=2, padding=0)                       # out: 1 x 1 x 1
        )

    # Build the neural block
    def make_disc_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            # Note: in critic (or discriminator in WGAN-GP model), we can use InstanceNorm2d instead of BatchNorm2d
            # nn.BatchNorm2d(output_channels),
            nn.InstanceNorm2d(output_channels, affine=True),    # ffine=True --> means that it has learnable parameters
            nn.LeakyReLU(0.2)
        )


    def forward(self, image):
        '''
            Forward function is to complete a forward pass of the critic: is given an image tensor,
            then returns a 1-dimension tensor representing fake/real.
            Parameters:
                       image: a flattened image tensor with dimension (im_chan)
                       '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

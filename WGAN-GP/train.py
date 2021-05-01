######### Training Initializations #########

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import Generator, Critic
from utils import show_images, gradient_penalty
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
""" 
    Training Initializations

    Setting the parameters:

        1- n_epochs: the number of times you iterate through the entire dataset when training
        2- z_dim: the dimension of the noise vector
        3- display_step: how often to display/visualize the images
        4- batch_size: the number of images per forward/backward pass
        5- lr: the learning rate
        6- beta_1, beta_2: the momentum terms
        7- lambda_GP: weight of the gradient penalty
        8- crit_iteration: number of times to update the critic per generator update - 
           there are more details about this in the Putting It All Together section
        9- device: the device type
"""

# Set the parameters
n_epochs = 100
z_dim = 100
display_step = 500
batch_size = 64
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
Lambda_GP = 10
crit_iteration = 5
device = "cuda" if torch.cuda.is_available() else "cpu"


##### Load ANIME_FACE dataset as tensors #####

IMG_SIZE = 64
IMG_CHANNELS = 3

path_images = "../input/Dataset_anime_faces"
# Create dataset
dataset = ImageFolder(root = path_images,
                      transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                                      transforms.CenterCrop(IMG_SIZE),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          [0.5 for _ in range(IMG_CHANNELS)],
                                                          [0.5 for _ in range(IMG_CHANNELS)]
                                                      )]))

# Create the dataloader
dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True)

"""Next, we initialize the generator, discriminator, and optimizers"""

gen = Generator(z_dim).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
opt_crit = optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

'''Initialize weights:
   Here, we want to initialize the weights to the normal distribution
   with mean 0 and standard deviation 0.02
'''
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.2)
    if isinstance(m , nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.2)
        nn.init.normal_(m.bias, 0)

gen = gen.apply(initialize_weights)
crit = crit.apply(initialize_weights)


######################### Train WGAN-GP ###############################

""" 
    Finally, we can train the WGAN-GP model! For each epoch, we will process the entire dataset in batches. 
    For every batch, we will update the discriminator and generator.

"""

####### Note #######

"""  WGAN-GP isn't necessarily meant to improve overall performance of a GAN, 
      but just **increases stability** and **avoids mode collapse**. 
      In general, a WGAN will be able to train in a much more stable way than the vanilla DCGAN, 
      though it will generally run a bit slower. Also, we  should train WGAN model for more epochs without it collapsing.
"""

step = 0
generator_losses = []
critic_losses = []

for epoch in range(n_epochs):
    # Dataloader returns the batches
    # (real, label) = (real, _)
    for real, _ in tqdm(dataloader):
        current_batch_size = len(real)
        real = real.to(device)


        #------------------
        #  Train Critic
        #------------------
        mean_iteration_critic_loss = 0
        for _ in range(crit_iteration):
            opt_crit.zero_grad()
            # Generate fake images
            noise = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            # the critic's scores of the fake images
            critic_fake = crit(fake.detach())
            # the critic's scores of the real images
            critic_real = crit(real)

            # Calculate gradient penalty
            # gp : the unweighted gradient penalty
            gp = gradient_penalty(crit, real, fake, device=device)
            crit_loss = torch.mean(critic_fake) - torch.mean(critic_real) + Lambda_GP * gp

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_iteration
            # Update gradient
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            opt_crit.step()
        critic_losses += [mean_iteration_critic_loss]



    #---------------------
    # Train Generator
    #---------------------
    opt_gen.zero_grad()
    # Generate fake image
    noise_2 = torch.randn(current_batch_size, z_dim, 1, 1).to(device)
    fake_2 = gen(noise_2)

    # Try to fool the discriminator model
    crit_fake_pred = crit(fake_2)
    # Generator loss
    gen_loss = -1. * torch.mean(crit_fake_pred)
    # Update gradients
    gen_loss.backward()
    # Update optimizer
    opt_gen.step()

    # Keep track of the average critic loss in this batch
    generator_losses += [gen_loss.item()]

    ################ Visualization result #################
    if step % display_step == 0 and step > 0:
        gen_mean = sum(generator_losses[-display_step:]) / display_step
        crit_mean = sum(critic_losses[-display_step:]) / display_step
        print(f"Step {step}: Generator loss {gen_mean}, Critic loss: {crit_mean}")
        show_images(fake)
        show_images(real)
        step_bins = 20
        n_examples = (len(generator_losses) // step_bins) * step_bins

        plt.plot(range(n_examples // step_bins),
        torch.Tensor(generator_losses[:n_examples]).view(-1, step_bins).mean(1),
        label = "Generator Loss")

        plt.plot(range(n_examples // step_bins),
        torch.Tensor(critic_losses[:n_examples]).view(-1, step_bins).mean(1),
        label = "Critic Loss")

        plt.legend()
        plt.show()
    step += 1
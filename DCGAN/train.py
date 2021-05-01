######### Training Initializations #########

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import Generator, Discriminator
from utils import show_images
from tqdm.auto import tqdm

""" Training parameters

      First, we shoud set the parameters including:
      
           1- criterion: the loss function
           2- n_epochs: the number of times you iterate through the entire dataset when training
           3- z_dim: the dimension of the noise vector
           4- display_step: how often to display/visualize the images
           5- batch_size: the number of images per forward/backward pass
           6- lr: the learning rate
           7- beta_1, beta_2: the momentum term
           8- device: the device type

      Second, we will load the dataset as tensors using a dataloader.
"""

# Set the parameters

# set the computation device
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
z_dim = 64
display_step = 500
batch_size = 32
lr = 3e-4
beta_1 = 0.5
beta_2 = 0.999
# Image size. All images will be resized to this size using a transformer.
IMG_SIZE = 64
# Number of channels in images during training. For color images the number is 3
IMG_CHANNEL = 3


##### Load ANIME_FACE dataset as tensors #####
'''
    Note: According to the rules of the DataLoader in pytorch you should choose the the superior path of the image path. 
    That means if your images locate in './Dataset/images/', the path of the data loader should be './Dataset' instead.
'''

path_images = "/content/drive/MyDrive/Dataset_anime_faces"
# Create the dataset
dataset = ImageFolder(root = path_images,
                      transform = transforms.Compose([
                          transforms.Resize(IMG_SIZE),
                          transforms.CenterCrop(IMG_SIZE),
                          transforms.ToTensor(),
                          # [0.5 for _ in range(IMG_CHANNEL)] = (0.5, 0.5, 0.5)
                          transforms.Normalize(
                              [0.5 for _ in range(IMG_CHANNEL)], [0.5 for _ in range(IMG_CHANNEL)]
                          )
                      ]))
# Create the dataloader
dataloader = DataLoader(dataset,
                        batch_size = batch_size,
                        shuffle = True)

"""Next, we initialize the generator, discriminator, and optimizers """

gen = Generator(z_dim).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device)
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# Here, we want to initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.normal_(m.bias, 0)

gen = gen.apply(initialize_weights)
disc = disc.apply(initialize_weights)

print(gen)
print(disc)


######################### Train DCGAN ###############################

""" 
    Finally, we can train the GAN model! For each epoch, we will process the entire dataset in batches. 
    For every batch, we will update the discriminator and generator. Then, we can see DCGAN's results!
"""

step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):
    # Dataloader returns the batches
    # (real, label) = (real, _)
    for real, _ in tqdm(dataloader):
        current_batch_size = len(real)
        real = real.to(device)

        ####################################
        #         Train Discriminator
        ####################################

        # Discriminator gradients
        opt_disc.zero_grad()

        # Generate fake images
        fake_noise = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
        fake_imgs = gen(fake_noise)

        ####### Pass fake images through discriminator #######
        # The detach() method constructs a new view on a tensor which is declared not to need gradients
        disc_fake_pred = disc(fake_imgs.detach())
        fake_targets = torch.zeros_like(disc_fake_pred)
        disc_fake_loss = criterion(disc_fake_pred, fake_targets)

        ####### Pass real images through discriminator #######
        disc_real_pred = disc(real)
        real_targets = torch.ones_like(disc_real_pred)
        disc_real_loss = criterion(disc_real_pred, real_targets)
        # Total loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        ####### Keep track of the average discriminator loss ########
        mean_discriminator_loss += disc_loss.item() / display_step

        ####### Update gradients ########
        disc_loss.backward(retain_graph=True)

        ####### Update optimizer ########
        opt_disc.step()

        ####################################
        #          Train Genrator
        ####################################

        # Generator gradients
        opt_gen.zero_grad()

        # Generate fake images
        fake_noise_2 = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
        fake_imgs_2 = gen(fake_noise_2)

        # Try to fool the discriminator model
        disc_fake_pred = disc(fake_imgs_2)
        targets = torch.ones_like(disc_fake_pred)
        gen_loss = criterion(disc_fake_pred, targets)
        # Update gradients
        gen_loss.backward()
        opt_gen.step()

        ####### Keep track of the average generator loss #######
        mean_generator_loss += gen_loss.item() / display_step

        ################ Visualization result #################
        if step % display_step == 0 and step > 0:
            print(f"Step {step}: Generator loss {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
            show_images(fake_imgs)
            show_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        step += 1


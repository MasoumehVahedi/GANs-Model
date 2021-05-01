######### Training Initializations #########

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator
from utils import vectors_embedding, one_hot_labels, input_dimension
from data_processing import crop_images
from tqdm.auto import tqdm
"""
    Setting the parameters:
      1- image_shape: the number of pixels in each DOG image, which has dimensions 64 x 64 and three channel so 3 x 64 x 64
      2- num_classes: the number of classes in DOG dataset (len(class_dict))
      3- criterion: the loss function
      4- n_epochs: the number of times you iterate through the entire dataset when training
      5- z_dim: the dimension of the noise vector
      6- display_step: how often to display/visualize the images
      7- batch_size: the number of images per forward/backward pass
      8- lr: the learning rate
      9- device: the device type
"""

image_shape = (3, 64, 64)
num_classes = 120

criterion = nn.BCEWithLogitsLoss()
n_epochs = 50
z_dim = 100
display_step = 500
batch_size = 128
lr = 0.0002
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create the dataset
train_data, names_only_dog_images, idxIn, labels = crop_images()
print(idxIn)
train_data = np.array(train_data)
train_data = train_data.transpose(0, 3, 2, 1)
print(train_data.shape)

labels = np.array(labels)
print(labels.shape)


# Create the dataset
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(labels))
dataloader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        pin_memory=True)


# Initialize the generator, discriminator, and optimizers
input_dim_generator, img_channel_discriminator = input_dimension(z_dim, image_shape, num_classes)

gen = Generator(input_dim = input_dim_generator).to(device)
disc = Discriminator(img_channels = img_channel_discriminator).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)

writer_real = SummaryWriter(f"D:/logs/real")
writer_fake = SummaryWriter(f"D:/logs/fake")

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


######################### Train CGAN ###############################
# Next to train CGAN that we want to like both the generator and the discriminator
# to know what class of image should be generated.

step = 0
generator_loss = []
discriminator_loss = []

#UNIT TEST NOTE: Initializations needed for grading
noise_and_labels = False
fake_imgs = False

fake_imgs_labels = False
real_imgs_labels = False
disc_fake_pred = False
disc_real_pred = False


for epoch in range(n_epochs):
    # Dataloader will return the batches and labels
    for real, labels in tqdm(dataloader):
        print(type(real))
        print(type(labels))
        print(real.shape)
        current_batch_size = len(real)
        print(current_batch_size)
        # Flatten the batch of real images
        real = real.to(device)
        labels = labels.to(device)

        oh_labels = one_hot_labels(labels.to(device), num_classes)
        oh_labels_image = oh_labels[:, :, None, None]
        print(oh_labels_image.shape)
        oh_labels_image = oh_labels_image.repeat(1, 1, image_shape[1], image_shape[2])
        print(oh_labels_image.shape)

        #=================================
        #     Train Discriminator
        #=================================
        # Discriminator gradients
        opt_disc.zero_grad()

        # Generate fake images
        fake_noise = torch.randn(current_batch_size, z_dim, device=device)
        print(fake_noise.shape)
        print(oh_labels.shape)
        # Now, we need to combine the noise vectors and the one-hot labels for the generator
        # Then, generate the conditioned fake images
        noise_and_labels = vectors_embedding(fake_noise, oh_labels)
        fake_imgs = gen(noise_and_labels)

        # In this step, we should get a prediction from the discriminator model:
        # First of all, we need to create the input for the discriminator:
        #    1- combining the fake images with oh_labels_image
        #    2- combining the real images with oh_labels_image
        fake_imgs_labels = vectors_embedding(fake_imgs, oh_labels_image)
        real_imgs_labels = vectors_embedding(real, oh_labels_image)
        # Secondly, we need to get prediction of the discriminator on the both fake images and real ones
        disc_fake_pred = disc(fake_imgs_labels.detach())  # to detach the generator (.detach()) so you do not backpropagate through it
        disc_real_pred = disc(real_imgs_labels)

        # Losses
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        disc_loss.backward(retain_graph=True)
        opt_disc.step()

        discriminator_loss += [disc_loss.item()]

        #===============================
        #       Train Generator
        #===============================
        # Generator gradients
        opt_gen.zero_grad()

        # Generate fake images
        fake_imgs_labels = vectors_embedding(fake_imgs, oh_labels_image)
        # we will give an error if we didn't concatenate our labels to our image correctly
        # Try to fool the discriminator model
        disc_fake_pred = disc(fake_imgs_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        # Update gradients
        gen_loss.backward()
        opt_gen.step()

        generator_loss += [gen_loss.item()]

        ################ Visualization result on tensorboard #################
        if current_batch_size % 100 == 0 and current_batch_size > 0:
            print(
                f"Epoch [{epoch}/{n_epochs}] Batch {current_batch_size}/{len(dataloader)} LossD: {disc_loss:.4f}, LossG: {gen_loss:.4f}"
            )
            with torch.no_grad():
                fake = gen(fake_noise, labels)
                # take out up to 32 example
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Rael", img_grid_real, global_step=step)
            step += 1
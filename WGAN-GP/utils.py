import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


""" 
    Gradient Penalty
    In this step, we calculate gradient_penalty function.
"""

def gradient_penalty(crit, real, fake, device="cpu"):
    '''This function will return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        '''
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.randn((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # Mix the images together: according to the formula in the original paper
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate critic scores
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores according to the images
    gradient = torch.autograd.grad(
        inputs = mixed_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True
    )[0]     # we take the first element of those

    # Now, we will reshape the gradient(flat it), so that each row captures one image
    gradient = gradient.view(gradient.shape[0], -1)

    # Calculate magnitude of each row
    gradient_norm = gradient.norm(2, dim=1)      # Here, we take the L2 norm

    # Gradient penalty
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty



''' 
    Function for visualizing images:
    convert a tensor to a numpy array
'''
def show_images(image_tensor, num_images=25, size=(3, 64, 64), nrow=5):
    image_tensor = (image_tensor + 1) / 2
    img_unflat = image_tensor.detach().cpu()
    img_grid = make_grid(img_unflat[:num_images], nrow=nrow)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()


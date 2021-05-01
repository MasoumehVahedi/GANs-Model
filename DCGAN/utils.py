
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



'''
   Function for visualizing images: Given a tensor of images, 
   number of images, 
   and size per image
'''
def show_images(image_tensor, num_img=25, size=(3, 64, 64), nrow=5):
    img_tensor = (image_tensor + 1) / 2
    img_unflat = img_tensor.detach().cpu()
    img_grid = make_grid(img_unflat[:num_img], nrow = nrow)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()
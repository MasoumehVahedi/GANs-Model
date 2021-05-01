###################### Load images and annotation ##########################

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET


"""
    About Dog File:
    
    1- all-dogs.zip - All dog images contained in the Stanford Dogs Dataset
    2- Annotations.zip - Class labels, Bounding boxes
"""


# Image size. All images will be resized to this size using a transformer.
IMG_SIZE = 64
# Number of channels in images during training. For color images the number is 3
IMG_CHANNEL = 3

path_images = "D:/dog dataset/all-dogs/"
path_annotation = "D:/dog dataset/Annotation/Annotation/"

all_images = os.listdir(path_images)

annotations = []
for a in glob.glob(path_annotation + "*"):
    annote = glob.glob(a + "/*")
    annotations += annote
# print(annotations)

# How many classes are there in all_dog
class_dict = {}
for line in annotations:
    key = line.split("\\")[-2]
    index = key.split("-")[0]
    class_dict.setdefault(index, key)
#num_classes = len(class_dict)

class_dict_2 = {}
for i,b in enumerate(class_dict.keys()):
    class_dict_2[b] = i

def bounding_box(img):
    bpath = path_annotation + str(class_dict[img.split('_')[0]])+'/'+str(img.split('.')[0])
    tree  = ET.parse(bpath)
    root  = tree.getroot()
    objects = root.findall('object')
    bbxs = []
    for o in objects:
        bndbox = o.find('bndbox') #reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbxs.append((xmin,ymin,xmax,ymax))
    return bbxs

new_imgs_filename = []
for img in all_images:
  bbx = bounding_box(img)
  for i,(b) in enumerate(bbx):
    new_imgs_filename.append(img[:-4]+'_'+str(i)+'.jpg')

# Here, using bounding box, we are going to crop images to get dogs only
def crop_images():
    path_images = "D:/dog dataset/all-dogs"
    path_annotation = "D:/dog dataset/Annotation/Annotation"
    IMAGES = os.listdir(path_images)
    annotations = os.listdir(path_annotation)
    idxIn = 0
    bbxs = []
    names_only_dog_images = []
    train_data = []

    labels = []
    for i, all_img_path in enumerate(new_imgs_filename):
        # img
        img_path = all_img_path[:-6] + '.jpg'
        # label
        label = class_dict_2[img_path.split('_')[0]]
        labels.append(label)

    # CROP WITH BOUNDING BOXES TO GET DOGS ONLY
    # iterate through each directory in annotation
    for annote in annotations:
        # iterate through each file in the directory
        for dog in os.listdir(os.path.join(path_annotation, annote)):
            try:
                img = Image.open(path_images + '/' + dog + '.jpg')
            except:
                continue
            # Element Tree library allows for parsing xml and getting specific tag values
            tree = ET.parse(path_annotation + '/' + annote + '/' + dog)
            # take a look at the print out of an xml previously to get what is going on
            root = tree.getroot()  # <annotation>
            objects = root.findall('object')  # <object>
            for o in objects:
                bndbox = o.find('bndbox')  # <bndbox>
                xmin = int(bndbox.find('xmin').text)  # <xmin>
                ymin = int(bndbox.find('ymin').text)  # <ymin>
                xmax = int(bndbox.find('xmax').text)  # <xmax>
                ymax = int(bndbox.find('ymax').text)  # <ymax>
                bbxs.append((xmin,ymin,xmax,ymax))
                w = np.min((xmax - xmin, ymax - ymin))
                img_2 = img.crop((xmin, ymin, xmin + w, ymin + w))
                img_2 = img_2.resize((64, 64), Image.ANTIALIAS)
                train_data.append(np.asarray(img_2))
                names_only_dog_images.append(annote)
                idxIn += 1

    return train_data, names_only_dog_images, idxIn, labels

train_data, names_only_dog_images, idxIn, labels = crop_images()
trainData = np.array(train_data)


# See the len of each files
print("Total images: {}".format(len(all_images)))
print("Total annotations: {}".format(len(annotations)))
print("Total classes: {}".format(len(class_dict)))
print("Total idxIn: {}".format(idxIn))


# Plotting some dog images after crop
r = np.random.randint(0, idxIn, 25)

for i in range(5):
    plt.figure(figsize=(12, 10))
    for j in range(5):
        plt.subplot(1, 5, j+1)
        img = Image.fromarray(trainData[r[i*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        plt.title(names_only_dog_images[r[i*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()


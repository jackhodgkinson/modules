# Functions to handle Image Data 

## Package Import
### Python Packages
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F

### Python modules
import random

### User Created Modules
import data_manip
from data_manip import get_var_name, dataset_namer
from file_path import file_path


## Functions 
### Function to convert tensor images to different colour 
def image_convert(dataset, mode=''):
    
    mode = mode.lower()
    
    if mode == 'gray':
            dataset.imgs = [Image.fromarray(np.uint8(img)) for img in dataset.imgs]
            dataset.imgs = [F.rgb_to_grayscale(img) for img in dataset.imgs]
            dataset.imgs = np.stack([np.array(img) for img in dataset.imgs])
    return dataset

### Function to extract features and labels from image data
def features_labels(dataset, method):  
    
    # Extract features and transform to torch
    X = dataset.imgs
    X = torch.from_numpy(X)

    # Extract labels and transform to torch
    y = dataset.labels
    y = torch.from_numpy(y) 
    
    # Output features and labels
    features = X
    labels = y
    
    return features, labels

### Function to generate random images from dataset
def image_generator(name,data,ID,colour_change, number):
    
    y = len(data.imgs)
    if number == '':
        number = random.randint(0,y-ID)
    elif number in range(0,y-ID):
        number 
    else:
        print("ERROR: x parameter must be blank or an integer between 0 and the number of features.")
    for i in range(number, number+ID): 
        numpy_array = data.imgs[i]
        image = Image.fromarray(np.uint8(numpy_array))
        filename = file_path+f"Results/Images/{name}_{colour_change.lower()}_sample{i}.jpeg"
        image.save(filename)
import os 
import glob
import torch

import numpy as np 
import pandas as pd

from PIL import Image, ImageFile

from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms 

from albumentations import (
    Compose,
    OneOf, 
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SIIMDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        imageIds, 
        transform=True,
        preprocessingFn=None
    ):
        """
        Dataset class for segmentation problem
        :param imageIds: ids of images, list
        :param transform: True/False, no transform in validation
        :param preprocessingFn: a function for preprocessing image
        """
        # create empty dict to store image and mask paths
        self.data = defaultdict(dict)
        # for augmentations
        self.transform = transform 
        # preprocessing function to normalize images
        self.preprocessingFn = preprocessingFn
        # albumentation augmentations: shift, scare and rotate
        # we apply with prob. 80%
        # one of gamma and brightness/contrast applied also
        self.aug = Compose(
            [
                ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.8
                ),
                OneOf(
                    [
                        RandomGamma(
                            gamma_limit=(90, 110)                        
                        ),
                        RandomBrightnessContrast(
                            brightness_limit=0.1,
                            contrast_limit=0.1
                        )                            
                    ],
                    p=0.5
                )
            ]
        )
        # going over all imageIds to store image and mask paths
        for imgId in imageIds:
            files = glob.glob(os.path.join(TRAIN_PATH, imgId, "*.png"))
            self.data[counter] = {
                "imgPath": os.path.join(
                    TRAIN_PATH, imgID + ".png"
                ),
                "maskPath": os.path.join(
                    TRAIN_PATH, imgId + "_mask.png"
                ),
            }
    def __len__(self):
        # return length of dataset
        return len(self.data)
    
    def __getitem__(self, item):
        # for a given item index, return image and mask tensors
        # and read image and mask paths
        imgPath = self.data[item]["imgPath"]
        maskPath = self.data[item]["maskPath"]

        # read image and convert to RGB
        img = Image.open(imgPath)
        img.convert("RGB")
        
        # PIL image to numpy array
        img = np.array(img)

        # read mask image and convert to binary float matrix
        mask = Image.open(maskPath)
        mask = (mask >= 1).astype("float32")

        # if in training data, apply transforms
        if self.transform is True:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        #preprocess tthe image using provided preprocessing tensors
        img = self.preprocessingFn(img)

        # return image and mask
        return {
            "image": transforms.ToTensor()(img),
            "mask": transforms.ToTensor()(mask).float,
        }
    
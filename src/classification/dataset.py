import torch
import numpy as np
from PIL import Image, ImageFile 

# take care of images without an ending bit (corrupt)
ImageFile.LOAD_TRUNCATED_IMAGES = True 

class ClassificationDataset:
    """ 
    A general classification dataset class that can be used for all kinds of 
    image classification problems.
    """
    def __init__(self,
                imagePaths,
                targets,
                resize=None,
                augmentations=None
                ):
        """
        :param imagePaths: list of paths to images
        :param targets: numpy array
        :param resize: tuple, e.g. (256, 256)
        :param augmentations: albumentation augmentations
        """
        self.imagePaths = imagePaths
        self.targets = targets 
        self.resize = resize 
        self.augmentations = augmentations 
        
    def __len__(self):
        """
        Return total number of samples in dataset 
        """
        return len(self.imagePaths)
    
    def __getitem__(self, item):
        """
        For a given item index, return everything we need to train a given model
        """
        # PIL to open image
        image = Image.open(self.imagePaths[item])
        # convert to RGB
        image = image.convert("RGB")
        # get correct targets
        targets = self.targets[item]
        # resize if needed
        if self.resize is not None:
            image = image.resize(
            (self.resize[1], self.resize[0]),
            resample=Image.BILINEAR
            )
        # convert image to numpy array 
        image = np.array(image)
        # add albumentation augmentations 
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        # pytorch expects CHW not HWC
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        # return tensors of images and targets (be careful with type: for regression we need dtype=torch.float)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
    
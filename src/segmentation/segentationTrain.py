import os 
import sys
import torch 

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

from apex import amp
from collections import OrderedDict
from sklearn import model_selection
from tqdm import tqdm
from torch.optim import lr_scheduler

from segmentationDataset import SIIMDataset
import config

# training csv filepath
TRAINING_CSV = config.TRAINING_CSV
# training and test batch sizes
TRAINING_BATCH_SIZE = config.TRAINING_BATCH_SIZE
TEST_BATCH_SIZE = config.TEST_BATCH_SIZE
# number of epochs 
EPOCHS = config.EPOCHS

# define the encoder for U-net: https://github.com/qubvel/segmentation_models.pytorch
ENCODER = config.ENCODER
ENCODER_WEIGHTS = config.ENCODER_WEIGHTS

# train on gpu
DEVICE = config.DEVICE

def train(dataset, dataLoader, model, criterion, optimizer):
    """
    Training function that trains for one epoch 
    :param dataset: dataset class (SIIMDataset)
    :param dataLoader: this is the pytorch dataloader
    :param model: model
    :param criterion: loss function
    :param optimizer: optimizer, e.g. adam, sgd, ...
    """
    # put model in train mode
    model.train()
    # calc. number of batches
    numBatches = int(len(dataset)/ dataLoader.batch_size)
    # initilize tqdm to track our progress
    tk0 = tqdm(dataLoader, total=numBatches)
    # loop over all batches
    for d in tk0:
        # fetch input and masks from dataset batch 
        inputs = d["image"]
        targets = d["mask"]
        # move inputs and targets to cpu/gpu device
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.float)
        # zero grad to optimizer 
        optimizer.zero_grad()
        # forward step 
        outputs = model(inputs)
        # calc. loss
        loss = criterion(outputs, targets)
        # backwards loss is calculated: we are using mixed precision training
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # step optimizer
        optimizer.step()
    # close tqdm
    tk0.close()

def evaluate(dataset, dataLoader, model):
    """
    This function does the evaluation for one epoch.
    :param dataset: dataset class (SIIMDataset)
    :param dataLoader: this is the pytorch dataloader
    :param model: pytorch model
    """
    # put model in eval mode 
    model.eval()
    # init final loss to 0 
    finalLoss = 0
    # calculate number of batches and initilize tqdm
    numBatches = int(len(dataset)/ dataLoader.batch_size)
    tk0 = tqdm(dataLoader, total=numBatches)
    # we use no_grad context to speed things up 
    with torch.no_grad():
        for d in tk0:
            inputs = d["image"]
            targets = d["mask"]
            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.float)
            output = model(inputs)
            loss = criterion(output, targets) 
            # add loss to final loss
            finalLoss += loss
    # close tqdm 
    tk0.close()
    # return average loss over all batches
    return finalLoss / numBatches

if __name__ == "__main__":
    # read the training csv file 
    df = pd.read_csv(TRAINING_CSV)
    # split data into train and valid
    dfTrain, dfValid = model_selection.train_test_split(
        df, random_state=42, test_size=0.1
    )
    # train and valid images lists/arrays
    trainingImages = dfTrain.image_id.values
    validationImages = dfValid.image_id.values
    # fetch UNet model from segmentation models
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=None
    )
    # we normalize images
    prepFn = smp.encoders.get_preprocessing_fn(
        ENCODER,
        ENCODER_WEIGHTS
    )
    # send model to device
    model.to(DEVICE)
    # init training dataset, with transform true
    trainDataset = SIIMDataset(
        trainingImages,
        transform=True,
        preprocessingFn=prepFn
    )
    # wrap in torch dataloader
    trainLoader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    # init validation dataset, aug is disabled
    validDataset = SIIMDataset(
        validationImages,
        transform=False,
        preprocessingFn=prepFn
    )
    # wrap in torch dataloader
    validLoader = torch.utils.data.DataLoader(
        validDataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    # define criterion 
    
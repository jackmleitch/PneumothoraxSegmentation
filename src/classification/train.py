import os 

import pandas as pd
import numpy as np 

import albumentations 
import torch 

from sklearn import metrics 
from sklearn.model_selection import train_test_split 

import dataset 
import engine 
from model import getModel 

if __name__ == '__main__':
    # locations of train.csv and train_png folder
    csvPath = '../input/siim-png-train-csv/train.csv'
    pngPath = '../input/siim-png-images/train_png/'
    
    # cuda device 
    device = 'cuda'
    
    # number of epochs we train for 
    epochs = 10
    
    # load in dataframe 
    df = pd.read_csv(csvPath)
    
    # fetch image Ids
    images = df.ImageId.values.tolist()
        
    # image location list 
    images = [
        os.path.join(pngPath, i + '.png') for i in images
    ]
            
    # binary targets to numpy array 
    targets = df.target.values
    
    # fetch our model, can use both T/F for pretrained 
    model = getModel(pretrained=True)
    
    # move model to device 
    model.to(device)
    
    # meand and std RGB values from imagenet dataset 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # we apply normalization 
    aug = albumentations.Compose(
        [albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)]
    )
    
    # train_test_split instead of kfold (just to try first model)
    trainImages, validImages, trainTargets, validTargets = train_test_split(
        images, targets, stratify=targets, random_state=42
    )
    
    # fetch ClassificationDataset class from dataset 
    trainDataset = dataset.ClassificationDataset(
        imagePaths=trainImages,
        targets=trainTargets,
        resize=(227,227),
        augmentations=aug
    )
    
    validDataset = dataset.ClassificationDataset(
        imagePaths=validImages,
        targets=validTargets,
        resize=(227,227),
        augmentations=aug
    )
    
    # create batches of data using torch dataloader
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=16, shuffle=True, num_workers=4
    )
    
    validLoader = torch.utils.data.DataLoader(
        validDataset, batch_size=16, shuffle=True, num_workers=4
    )
    
    # ADAM optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # train and print auc score for all epochs
    for epoch in range(epochs):
        engine.train(trainLoader, model, optimizer, device=device)
        preds, validTargets = engine.evaluate(
            validLoader, model, device=device
        )
        rocAUC = metrics.roc_auc_score(validTargets, preds)
        print(
            f"Epoch={epoch}, Valid ROC AUC={rocAUC}"
        )
    
# Epoch=0, Valid ROC AUC=0.5759235999124819
# Epoch=1, Valid ROC AUC=0.45915739487694784
# Epoch=2, Valid ROC AUC=0.45988266087534335
# Epoch=3, Valid ROC AUC=0.6185878787387665
# Epoch=4, Valid ROC AUC=0.5964611881396724
# Epoch=5, Valid ROC AUC=0.6260325924005088
# Epoch=6, Valid ROC AUC=0.6289547255739326
# Epoch=7, Valid ROC AUC=0.582583891801658
# Epoch=8, Valid ROC AUC=0.6730671053377956
# Epoch=9, Valid ROC AUC=0.7016174647293827
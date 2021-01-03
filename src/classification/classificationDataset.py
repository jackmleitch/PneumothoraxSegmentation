import os 
import numpy as np
import pandas as pd 

from PIL import Image 
from sklearn import ensemble, metrics, model_selection 
from tqdm import tqdm 

def createDataset(trainingDf, imageDir):
    """
    This function takes the training dataframe and outputs training array and labels.
    :param training_df: dataframe with image ID, target column 
    :param image_dir: location of images (folder), string 
    :return: X, y (training array with features and the corresponding lables)
    """
    # create empty list to store image vectors
    images = []
    # create empty list to store targets
    targets = []
    
    # loop over the dataframe
    for index, row in tqdm(
        trainingDf.iterrows(),
        total=len(trainingDf),
        desc="processing images"
    ):
        # get image ID
        imageID = row['ImageId']
        # create image path 
        imagePath = os.path.join(imageDir, imageID)
        # open image using PIL
        image = Image.open(imagePath + '.png')
        # resize image to 256x256 using bilinear resampling
        image = image.resize((256,256), resample=Image.BILINEAR)
        # convert image to array 
        image = np.array(image)
        # ravel 
        image = image.ravel()
        # append to images and target lists 
        images.append(image)
        targets.append(int(row['target']))
    # convert list of list of images to numpy array
    images = np.array(images)
    # print size of this array 
    print(images.shape)
    return images, targets 

if __name__ == "__main__":
    csvPath = '../input/siim-png-train-csv/train.csv'
    imagePath = '../input/siim-png-images/train_png/'

    # read CSV with image id and target columns 
    df = pd.read_csv(csvPath)

    # create new kfold column 
    df['kfold'] = -1
    # randomise rows of data
    df = df.sample(frac=1).reset_index(drop=True)
    # fetch labels 
    y = df.target.values
    # init kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill in kfold column 
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    for fold in range(5):
        # temp train and test dataframes
        trainDf = df[df.kfold != fold].reset_index(drop=True)
        testDf = df[df.kfold == fold].reset_index(drop=True)
        # create train/test dataset
        xtrain, ytrain = createDataset(trainDf, imagePath)
        xtest, ytest = createDataset(testDf, imagePath)
        
        # fit random forest model 
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)
        
        # predict prob of class 1
        preds = clf.predict_proba(xtest)[:, 1]
        
        # print results
        print(f"FOLD: {fold}")
        print(f"AUC = {metrics.roc_auc_score(ytest, preds)}")
        print("")

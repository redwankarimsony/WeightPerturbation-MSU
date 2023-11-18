import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision.models import DenseNet161_Weights
from tqdm import tqdm

from dataset import LivDetIris2020
from search_results_exhaustive_DenseNet import *

weight_key = "0.9114"
device = torch.device("cuda:1")

# model definition and weight loading
model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)

weights = torch.load(os.path.join(f"Results/LivDet-Iris-2020/GaussianNoise/DenseNet161GaussianNoise-{weight_key}-.pth"))
model.load_state_dict(weights)
model.to(device)
model.eval()

# Loading the dataset
livDetIris20_ds = LivDetIris2020(imageFolder="Iris_Image_Database",
                                 splitPath="Data-Splits/LivDet-Iris-2017/test_split-Seg.csv")
print(len(livDetIris20_ds))
livDetIris20_dl = DataLoader(livDetIris20_ds, batch_size=16, shuffle=False,
                             num_workers=int(os.cpu_count() * 0.5))

# Generating Predictions
testImgNames = []
testTrueLabels = []
results = []
for batch_idx, datasets in enumerate(tqdm(livDetIris20_dl, desc=f"Generating Prediction:")):
    data, imgName, label = datasets
    testImgNames.extend(imgName)
    testTrueLabels.extend(list(label.numpy()))
    data = data.to(device)
    predictions = model(data).detach().cpu().numpy()[:, 1]
    results.extend(predictions)

# Result Calculations:
predict_score = np.array(results)
predictScore = (predict_score - min(predict_score)) / (max(predict_score) - min(predict_score))
(fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)

# Calculating TDR at 0.2% FDR
TDR = np.interp(0.002, fprs, tprs)
print("True Detection Rate:", TDR)

# Saving Results.
with open(f"Results/LivDet-Iris-2020/GaussianNoise/{weight_key[2:]}-original.pkl", "wb") as f:
    pickle.dump([testTrueLabels, predictScore], f)

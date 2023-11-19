import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from torchvision.models import (DenseNet161_Weights,
                                ResNet101_Weights,
                                VGG19_BN_Weights)
from tqdm import tqdm

__all__ = ['kernelFusionDenseNet161',
           'kernelFusionVGG19',
           'kernelFusionResNet101',
           'loadNewModel',
           'loadSummary',
           'getOriginalLabels',
           'getPrediction',
           'getScore']


def kernelFusionVGG19(model1, model2, layer, ):
    layers = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49']

    for [name, param], [_, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        if (layer is None and (any(number in name for number in layers) or 'classifier' in name)) or (
                layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().cpu().numpy()
                weights2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((weights + weights2) / 2., dtype=torch.float))
            elif 'bias' in name:
                # print(f"Layer idx: {idx}, Name: {name}")
                # idx+=1
                bias = param.detach().cpu().numpy()
                bias2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((bias + bias2) / 2., dtype=torch.float))
    return model1


def kernelFusionResNet101(model1, model2, layer, ):
    for [name, param], [_, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        if (layer is None and ('conv' in name or 'fc' in name)) or (layer is not None and layer in name):
            if "weight" in name:

                weights = param.detach().cpu().numpy()
                weights2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((weights + weights2) / 2., dtype=torch.float))

            elif 'bias' in name:
                bias = param.detach().cpu().numpy()
                bias2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((bias + bias2) / 2., dtype=torch.float))
    return model1


def kernelFusionDenseNet161(model1, model2, layer, ):
    for [name, param], [_, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        if ((layer is None and ('conv' in name or 'classifier' in name)) or
                (layer is not None and layer in name)):
            if "weight" in name:
                weights = param.detach().cpu().numpy()
                weights2 = param2.detach().cpu().numpy()
                param.data = nn.Parameter(torch.tensor((weights + weights2) / 2., dtype=torch.float))

            elif 'bias' in name:
                bias = param.detach().cpu().numpy()
                bias2 = param2.detach().cpu().numpy()
                param.data = nn.Parameter(torch.tensor((bias + bias2) / 2., dtype=torch.float))
    return model1


def loadNewModel(desc=None, savedPath=None, device="cpu"):
    # Defining models
    if desc == 'DenseNet161':
        model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif desc == 'ResNet101':
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif desc == 'VGG19':
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        model.classifier.add_module('6', nn.Linear(4096, 2))
    else:
        raise ValueError("Model not defined.")

    # Loading weights/state dictionary
    model.load_state_dict(torch.load(savedPath))
    model.to(device)
    model.eval()
    return model


def loadSummary(csv_file: os.path):
    if str(csv_file).endswith(".csv"):
        df = pd.read_csv(csv_file)
        return df


def getOriginalLabels(dl: torch.utils.data.DataLoader):
    testImgNames, testTrueLabels = [], []
    for batch_idx, datasets in enumerate(dl):
        _, imgName, label = datasets
        testImgNames.extend(imgName)
        testTrueLabels.extend(label)

    torch.cuda.empty_cache()
    return testImgNames, testTrueLabels


def getPrediction(model: nn.Sequential, dataLoader: DataLoader, device: torch.device, message: str):
    # Generating predictions
    results = []
    model.to(device=device)
    with torch.no_grad():
        for batch_idx, datasets in enumerate(tqdm(dataLoader, desc=f"Predictions {message}")):
            data, imgName, label = datasets
            data = data.to(device)
            predictions = model(data).detach().cpu().numpy()[:, 1]
            results.extend(predictions)
    return np.array(results)


def getScore(rawScores, testTrueLabels, threshold=0.002):
    predictScore = (rawScores - min(rawScores)) / (max(rawScores) - min(rawScores))
    (fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)
    return np.interp(threshold, fprs, tprs)

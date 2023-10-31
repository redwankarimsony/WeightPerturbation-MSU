from email import message
from tkinter.messagebox import NO
import torch
import torchvision.models as models
from torchvision.models import DenseNet161_Weights
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import math
import os
import glob
import csv
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import roc_curve
import copy
import pickle
from numpy import linalg
import random
import time
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import LivDetIris2020
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
from search_results_exhaustive import *


def kernelFusionVGG19(model1, model2, layer,):
    layers = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49']

    for [name, param], [name2, param2] in zip(model1.named_parameters(), model2.named_parameters()):
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




def kernelFusionResNet101(model1, model2, layer,):
    idx=1
    for [name, param], [name2, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        if (layer is None and ('conv' in name or 'fc' in name)) or (layer is not None and layer in name):
            if "weight" in name:
                # print(f"Layer idx: {idx}, Name: {name}")
                # idx+=1
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



def kernelFusionDenseNet161(model1, model2, layer, ):
    idx = 1
    for [name, param], [name2, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        

        if (layer is None and ('conv' in name or 'classifier' in name)) or (layer is not None and layer in name):
            

            if "weight" in name:
                # print(f"Layer idx: {idx}, Name: {name}")
                # idx+=1
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-perturbation', default='GaussianNoise', type=str,
                        help='GaussianNoise, WeightsZero, WeightsScaling, TopWeightsZero, BottomWeightsZero, WeightsZeroScaling, Quantize, FiltersZero')
    parser.add_argument('-perturbationSetup', default='Entire', type=str, help='Entire, Layers')
    parser.add_argument('-scales', default='0.1', type=str, help='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('-splitPath', default='Data-Splits/LivDet-Iris-2020/test_split-Seg.csv', type=str)
    parser.add_argument('-imageFolder', default='Iris_Image_Database/', type=str)
    parser.add_argument('-modelPath', default='Model/LivDet-Iris-2020/DesNet161_best.pth', type=str)
    parser.add_argument('-resultPath', default='Results/LivDet-Iris-2020/', type=str)
    parser.add_argument('-model', default='ResNet101', type=str, help='DenseNet161, ResNet101, VGG19')
    parser.add_argument('-bestTDR', default=0.8411, type=int)
    parser.add_argument('-nmodels', default=1, type=int)
    args = parser.parse_args()

    # CUDA Device assignment.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Creating result directory
    resultPath = os.path.join(args.resultPath, args.perturbation, args.model)
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    # Defining scales
    scales = [float(scale) for scale in args.scales.split(',')]

    # Defining perturbations
    # Selectin layers based on the selected models
    if args.perturbationSetup == 'Entire':
        layers = [None]
    else:
        if args.model == 'DenseNet161':
            layers = ['features.denseblock4.denselayer24.conv2', 
                        'features.denseblock3.denselayer36.conv2',
                        'features.denseblock2.denselayer12.conv2',
                        'features.denseblock1.denselayer6.conv2',
                        'features.conv0']
        elif args.model == 'ResNet101':
            layers = ['conv1',
                      'layer1.2.conv3.weight',
                      'layer2.3.conv3.weight',
                      'layer3.22.conv3.weight',
                      'layer4.2.conv3.weight',
                      'fc.weight']
        elif args.model == 'VGG19':
            layers = ['classifier.6.weight',
                      'features.50.weight',
                      'features.40.weight',
                      'features.30.weight',
                      'features.20.weight',
                      'features.10.weight',
                      'features.0.weight']


    # Loading the dataset
    livDetIris20_ds = LivDetIris2020(imageFolder=args.imageFolder,
                                     splitPath=args.splitPath)
    livDetIris20_dl = DataLoader(livDetIris20_ds,
                                 batch_size=16,
                                 shuffle=False,
                                 num_workers=int(os.cpu_count() * 0.5))

    

    # Creating modelA and modelB
    # modelA = os.path.join(args.resultPath, args.perturbation, "VGG19GaussianNoise-0.8856-.pth")
    # modelB = os.path.join(args.resultPath, args.perturbation, "VGG19GaussianNoise-0.9031-.pth")
    modelA = "Results/LivDet-Iris-2020/GaussianNoise/DenseNet161GaussianNoise-0.9476-.pth"
    modelB = "Results/LivDet-Iris-2020/features_denseblock1_denselayer6_conv2/BottomWeightsZero-0.9686-.pth"
    modelA = loadNewModel(desc=args.model, device=torch.device("cuda:1"), savedPath=modelA)
    modelB = loadNewModel(desc=args.model, device=torch.device("cuda:1"), savedPath=modelB)
    modelA.eval()
    modelB.eval()




    # Loading the dataset
    livDetIris17_ds = LivDetIris2020(imageFolder=args.imageFolder,
                                        # splitPath='Data-Splits/LivDet-Iris-2020/test_train_split06-Seg.csv')
                                     splitPath=args.splitPath)
    livDetIris17_dl = DataLoader(livDetIris17_ds,
                                 batch_size=16,
                                 shuffle=False,
                                 num_workers=int(os.cpu_count() * 0.5))




    # Loading Ground Truth
    testImgNames, testTrueLabels = getOriginalLabels(livDetIris17_dl)

    # Performing Kernel Fusion
    if args.model == "DenseNet161":
        model_Kfused = kernelFusionDenseNet161(modelA, modelB, layer=None)
    elif args.model == "ResNet101":
        model_Kfused = kernelFusionResNet101(modelA, modelB, layer=None)
    elif args.model =="VGG19":
        model_Kfused = kernelFusionVGG19(modelA, modelB, layer=None)


    model_Kfused.eval()
    # print(model_Kfused)


    modelA_preds = getPrediction(model=modelA, dataLoader=livDetIris17_dl, device=torch.device("cuda:1"), message="modelA")
    modelB_preds = getPrediction(model=modelB, dataLoader=livDetIris17_dl, device=torch.device("cuda:1"), message="modelB")
    modelB = None
    modelA = None
    torch.cuda.empty_cache()
    model_Kfused_preds = getPrediction(model=model_Kfused, dataLoader=livDetIris17_dl, device=torch.device("cuda:1"), message="KFusion")
    model_Kfused = None
    torch.cuda.empty_cache()

    time.sleep(1)
    # print(modelA_preds.shape, modelB_preds.shape, model_Kfused_preds.shape)
    modelAScore = getScore(testTrueLabels=testTrueLabels, rawScores = modelA_preds, threshold=0.002)
    modelBScore = getScore(testTrueLabels=testTrueLabels, rawScores = modelB_preds, threshold=0.002)

    regFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores = (modelA_preds+modelB_preds)/2, threshold=0.002)
    kernelFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores =model_Kfused_preds, threshold=0.002)
    print(f"Perturbed Scores: {modelAScore, modelBScore}")
    print(f"Reg fusion: {regFusionScore}, Selective kernel fusion: {kernelFusionScore}\n\n\n")



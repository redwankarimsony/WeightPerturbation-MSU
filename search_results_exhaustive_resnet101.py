import torch
import torchvision.models as models
from torchvision.models import DenseNet161_Weights, ResNet101_Weights, VGG19_BN_Weights
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
    for [name, param], [name2, param2] in zip(model1.named_parameters(), model2.named_parameters()):
        if (layer is None and ('conv' in name or 'classifier' in name)) or (layer is not None and layer in name):
            if "weight" in name:
                weights = param.detach().cpu().numpy()
                weights2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((weights + weights2) / 2., dtype=torch.float))

            elif 'bias' in name:
                bias = param.detach().cpu().numpy()
                bias2 = param2.detach().cpu().numpy()
                param.data = torch.nn.Parameter(torch.tensor((bias + bias2) / 2., dtype=torch.float))
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
    # Loading weights/state dictionary
    model.load_state_dict(torch.load(savedPath))
    return model


def loadSummary(csv_file: os.path):
    if str(csv_file).endswith(".csv"):
        df = pd.read_csv(csv_file)
        return df

def getOriginalLabels(dl:torch.utils.data.DataLoader):
    testImgNames, testTrueLabels = [], []
    for batch_idx, datasets in enumerate(dl):
        _ , imgName, label = datasets
        testImgNames.extend(imgName)
        testTrueLabels.extend(label)

    torch.cuda.empty_cache()
    return testImgNames, testTrueLabels


def getPrediction(model:torch.nn.Sequential, dataLoader:torch.utils.data.DataLoader, device:torch.device, message:str):
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

def getScore(rawScores, testTrueLabels, threshold = 0.002):
    predictScore = (rawScores - min(rawScores)) / (max(rawScores) - min(rawScores))
    (fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)
    return np.interp(threshold, fprs, tprs)

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

    saved_weights = glob.glob(os.path.join(args.resultPath, args.perturbation, "*.pth"))
    # print(saved_weights)



    combinations = loadSummary(os.path.join(args.resultPath, args.perturbation, "summary-resnet101.csv"))
    print(combinations.head())
    predictionsPath = "/home/sonymd/Desktop/WeightPerturbation-MSU/Results/LivDet-Iris-2020/predictions-resnet101.pkl"
    
    pkl_file = open(os.path.join(args.resultPath, args.perturbation, "predictions-resnet101.pkl"), "rb")
    predictions = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open(os.path.join(args.resultPath, args.perturbation, "predictions-resnet101.pkl"), "wb")

    for rowIdx, row in combinations.iterrows():
        if combinations.iloc[rowIdx, 4] =="-" or combinations.iloc[rowIdx, 5] =="-":
            print("\n\n Row ID: ",rowIdx, "\n", row)
            # modelA, modelB, modelA-Score, modelB-Score, RegularFusion, KernelFusion

            # Creating modelA and modelB
            modelA = os.path.join(args.resultPath, args.perturbation, row["modelA"])
            modelB = os.path.join(args.resultPath, args.perturbation, row["modelB"])
            modelA = loadNewModel(desc=args.model, device=torch.device("cpu"), savedPath=modelA)
            modelB = loadNewModel(desc=args.model, device=torch.device("cpu"), savedPath=modelB)
            modelA.eval()
            modelB.eval()
            
            # Loading Ground Truth
            testImgNames, testTrueLabels = getOriginalLabels(livDetIris20_dl)

            # Performing Kernel Fusion
            model_Kfused = kernelFusionResNet101(modelA, modelB, layer=None)
            model_Kfused.eval()

            # p = Process(target=getPrediction, args=([modelA, LivDetIris2020, torch.device("cuda:1")], [modelB, LivDetIris2020, torch.device("cuda:0")],))
            # p.start()
            # p.join()
            if row["modelA"] in predictions.keys():
                modelA_preds = predictions[row["modelA"]]
                print(row["modelA"], "Already Calculated!")
            else:
                modelA_preds = getPrediction(model=modelA, dataLoader=livDetIris20_dl, device=torch.device("cuda:1"), message="modelA")
                predictions[row["modelA"]] = modelA_preds

            if row["modelB"] in predictions.keys():
                modelB_preds = predictions[row["modelB"]]
                print(row["modelB"], "Already Calculated!")
            else:
                modelB_preds = getPrediction(model=modelB, dataLoader=livDetIris20_dl, device=torch.device("cuda:0"), message="modelB")
                predictions[row["modelB"]] = modelB_preds
            pickle.dump(predictions, pkl_file)
            modelB = None
            modelA = None
            torch.cuda.empty_cache()
            model_Kfused_preds = getPrediction(model=model_Kfused, dataLoader=livDetIris20_dl, device=torch.device("cuda:1"), message="KFusion")
            model_Kfused = None
            torch.cuda.empty_cache()

            time.sleep(1)
            print(modelA_preds.shape, modelB_preds.shape, model_Kfused_preds.shape)
            regFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores = (modelA_preds+modelB_preds)/2, threshold=0.002)
            kernelFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores =model_Kfused_preds, threshold=0.002)
            print(f"reg fusion: {regFusionScore}, kernel fusion: {kernelFusionScore}")

            combinations.iloc[rowIdx, 4] = regFusionScore
            combinations.iloc[rowIdx, 5] = kernelFusionScore

            combinations.to_csv(os.path.join(args.resultPath, args.perturbation, "summary-resnet101.csv"), index=False)

        else:
            print(f"Row number {rowIdx+1} already calculated")





        
'''











    cross_match = {}
    for idxA, modelA in enumerate(saved_weights):
        for idxB, modelB in enumerate(saved_weights):
            if modelA == modelB:
                continue
            elif idxA < idxB:
                # Defining models
                model = loadNewModel(desc="DenseNet161",
                                     device=device,
                                     savedPath=modelA)
                model2 = loadNewModel(desc="DenseNet161",
                                      device=device,
                                      savedPath=modelB)
                model.eval()
                model2.eval()

                testData = pd.read_csv(args.splitPath, header=None)
                testPredScores = np.zeros((1, len(testData.values)))
                testTrueLabels = []
                testImgNames = []
                segInfo = None

                model_fused = kernelFusionDenseNet161(model, model2, layer=None)
                model = None
                model2 = None
                torch.cuda.empty_cache()
                model_fused.to(device=device)
                model_fused.eval()

                # Generating predictions
                results = []
                with torch.no_grad():
                    for batch_idx, datasets in enumerate(tqdm(livDetIris20_dl, desc=f"Generating Predictions:")):
                        data, imgName, label = datasets
                        testImgNames.extend(imgName)
                        testTrueLabels.extend(label)
                        data = data.to(device)
                        predictions = model_fused(data).detach().cpu().numpy()[:, 1]
                        results.extend(predictions)
                    testPredScores[0] = np.array(results)


                # Evaluation
                # Normalization of scores in [0,1] range
                predict_score = testPredScores[0]
                predictScore = (predict_score - min(predict_score)) / (max(predict_score) - min(predict_score))
                (fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)
                # imgNameScore = []
                # for i in range(len(testImgNames)):
                #     imgNameScore.append([testImgNames[i], testTrueLabels[i], predictScore[i]])

                # Calculating TDR at 0.2% FDR
                TDR = np.interp(0.002, fprs, tprs)
                print(TDR, modelA.split("/")[-1], modelB.split("/")[-1])

                cross_match[(modelA, modelB)] = [testImgNames, predictScore, testTrueLabels, (fprs, tprs, thresholds)]

                with open(resultPath + args.perturbation + '-fusions.pickle', 'wb') as f:
                    pickle.dump(cross_match, f)

'''
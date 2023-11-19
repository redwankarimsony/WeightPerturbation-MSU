import torch
import torchvision.models as models
from torchvision.models import DenseNet161_Weights
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import math
import os
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
from perturbation import weightPertubationDenseNet161, weightPertubationResNet101, weightPertubationVGG19
from dataset import preprocess_image, valTransform
from config import bestTDRs, base_model_paths, get_layers
from utils import get_cuda_device
from fusions import loadNewModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-perturbation', default='GaussianNoise', type=str,
                        help='GaussianNoise, WeightsZero, WeightsScaling, TopWeightsZero, '
                             'BottomWeightsZero, WeightsZeroScaling, Quantize, FiltersZero')
    parser.add_argument('-perturbationSetup', default='Entire', type=str, help='Entire, Layers')
    parser.add_argument('-scales', default='0.8, 0.2', type=str, help='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('-splitPath', default='Data-Splits/LivDet-Iris-2020/test_split-Seg.csv', type=str)
    parser.add_argument('-imageFolder', default='Iris_Image_Database/', type=str)
    parser.add_argument('-modelPath', default='Model/LivDet-Iris-2020/DesNet161_best.pth', type=str)
    parser.add_argument('-resultPath', default='Results/LivDet-Iris-2020/', type=str)
    parser.add_argument('-model', default='DenseNet161', type=str, help='DenseNet161, ResNet101, VGG19')
    parser.add_argument('-bestTDR', default=0.9022, type=int)
    parser.add_argument('-nmodels', default=1, type=int)
    args = parser.parse_args()

    # CUDA Device assignment.
    device = get_cuda_device()

    # Selecting the best TDR for the model and setting the base model path
    args.bestTDR = bestTDRs[args.model]
    args.modelPath = base_model_paths[args.model]
    print(f"Experiment: \nModel: {args.model}\nModelPath: {args.modelPath}")
    print(f"\nDataset: {args.splitPath} BestTDR: {args.bestTDR}\nDevice: {args.device}")


    # Creating result directory
    resultPath = os.path.join(args.resultPath, args.perturbation, args.model)
    os.makedirs(resultPath, exist_ok=True)

    # Defining scales
    scales = [float(scale) for scale in args.scales.split(',')]

    # Defining perturbations and selecting layers based on the selected models
    layers = get_layers(model=args.model, perturbationSetup=args.perturbationSetup)

    for i in range(0, args.nmodels):
        for layer in layers:

            # Defining models
            model = loadNewModel(args.model, args.modelPath)

            if layer is not None:
                resultPath = args.resultPath + layer.replace('.', '_') + '/'
                os.makedirs(resultPath, exist_ok=True)

            modelList, relChange = [], []

            for scale in scales:
                modelTemp = copy.deepcopy(model)

                # Perturbing models
                if args.model == 'DenseNet161':
                    modelTemp = weightPertubationDenseNet161(modelTemp, layer, args.perturbation, scale)
                elif args.model == 'ResNet101':
                    modelTemp = weightPertubationResNet101(modelTemp, layer, args.perturbation, scale)
                elif args.model == 'VGG19':
                    modelTemp = weightPertubationVGG19(modelTemp, layer, args.perturbation, scale)

                # Saving of Perturbed Model
                # states = {'state_dict': modelTemp.state_dict()}
                # torch.save(states, resultPath+ 'D-NetPAD_0.1.pth')

                # Calculating overall relative difference in the parameters
                diffParameters = torch.cat([(param_1 - param_2).view(-1) for param_1, param_2 in
                                            zip(modelTemp.parameters(), model.parameters())], dim=0)
                orgParameters = torch.cat([param_2.view(-1) for param_2 in model.parameters()], dim=0)
                relChange.append(
                    linalg.norm(diffParameters.detach().numpy()) / linalg.norm(orgParameters.detach().numpy()))
                modelTemp = modelTemp.to(device)
                modelTemp.eval()
                modelList.append(modelTemp)

            testData = pd.read_csv(args.splitPath, header=None)
            print("Number of models", len(modelList))
            testPredScores = np.zeros((len(modelList), len(testData.values)))
            testTrueLabels = []
            testImgNames = []
            segInfo = None
            for count, v in enumerate(tqdm(testData.values)):

                imageName = v[2]
                imagePath = args.imageFolder + imageName
                segInfo = v[3:]

                # Segmentation of image
                tranformImage, isImageSeg = preprocess_image(imagePath, segInfo)
                if isImageSeg:
                    tranformImage = tranformImage.unsqueeze(0)
                    tranformImage = tranformImage.to(device)

                    # Computing output from the models

                    for index, model in enumerate(modelList):
                        output = model(tranformImage[:, 0:3, :, :])
                        output = output.detach().cpu().numpy()[:, 1]
                        testPredScores[index, count] = output[0]

                    testImgNames.append(imageName)
                    if v[1] == 'Live':
                        testTrueLabels.append(0)
                    else:
                        testTrueLabels.append(1)

            with open(resultPath + args.perturbation + '.pickle', 'wb') as f:
                pickle.dump([testImgNames, testPredScores, testTrueLabels], f)

            for index, predict_score in enumerate(testPredScores):

                # Normalization of scores in [0,1] range
                predictScore = (predict_score - min(predict_score)) / (max(predict_score) - min(predict_score))
                (fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)
                imgNameScore = []
                for i in range(len(testImgNames)):
                    imgNameScore.append([testImgNames[i], testTrueLabels[i], predictScore[i]])

                # Calculating TDR at 0.2% FDR
                TDR = np.interp(0.002, fprs, tprs)

                # Storing results
                if TDR > args.bestTDR:
                    with open(resultPath + 'ImprovedTDRs-' + args.perturbation + '.csv', mode='a+') as fout:
                        fout.write("%s,%f,%f,%f\n" % (layer, scales[index], TDR, relChange[index]))
                with open(resultPath + 'TDR-' + args.perturbation + '.csv', mode='a+') as fout:
                    fout.write("%f,%f,%f\n" % (scales[index], TDR, relChange[index]))
                print("TDR @ 0.002 FDR with %s : %f @ %f \n" % (args.perturbation, TDR, scales[index]))

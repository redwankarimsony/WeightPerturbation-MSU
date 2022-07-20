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

from dataset import LivDetIris2020
from torch.utils.data import DataLoader

valTransform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

def preprocess_image(imgFile, SegInfo):
    isImageSeg = False
    try:
        image = Image.open(imgFile)

        # Segmentation and cropping of an image
        transform_img = []
        if len(SegInfo) >= 3:
            min_x = math.floor(int(SegInfo[0]) - int(SegInfo[2]) - 5)
            min_y = math.floor(int(SegInfo[1]) - int(SegInfo[2]) - 5)
            max_x = math.floor(int(SegInfo[0]) + int(SegInfo[2]) + 5)
            max_y = math.floor(int(SegInfo[1]) + int(SegInfo[2]) + 5)
            image = image.crop([min_x, min_y, max_x, max_y])

            transform_img = valTransform(image)
            transform_img = transform_img.repeat(3, 1, 1)
            isImageSeg = True
    except:
        isImageSeg = False
    return transform_img, isImageSeg


def weightPertubationDenseNet161(model, layer, perturbation, proportion):
    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and ('conv' in name or 'classifier' in name)) or (layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().cpu().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'TopWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), -K)[-K:]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), K)[:K]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                param.data = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))


            elif 'bias' in name:
                bias = param.detach().cpu().numpy()
                if perturbation == 'GaussianNoise':
                    bias = bias + np.random.normal(loc=0.0, scale=proportion * np.std(bias), size=bias.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'TopWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), -K)[-K:]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), K)[:K]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    bias = bias * proportion
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                    bias = bias * 5
                param.data = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    return model


def weightPertubationResNet101(model, layer, perturbation, proportion):
    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and ('conv' in name or 'fc' in name)) or (layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), -K)[-K:]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), K)[:K]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                param.data = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))
            elif 'bias' in name:
                bias = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    bias = bias + np.random.normal(loc=0.0, scale=proportion * np.std(bias), size=bias.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    bias = bias * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), -K)[-K:]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), K)[:K]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                    bias = bias * 5
                param.data = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    return model


def weightPertubationVGG19(model, layer, perturbation, proportion):
    layers = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49']

    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and (any(number in name for number in layers) or 'classifier' in name)) or (
                layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), -K)[-K:]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), K)[:K]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                param.data = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))
            elif 'bias' in name:
                bias = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    bias = bias + np.random.normal(loc=0.0, scale=proportion * np.std(bias), size=bias.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    bias = bias * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), -K)[-K:]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), K)[:K]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                    bias = bias * 5
                param.data = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    return model



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
    parser.add_argument('-model', default='DenseNet161', type=str, help='DenseNet161, ResNet101, VGG19')
    parser.add_argument('-bestTDR', default=0.9022, type=int)
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
            layers = ['features.denseblock4.denselayer24.conv2', 'features.denseblock3.denselayer36.conv2',
                      'features.denseblock2.denselayer12.conv2', 'features.denseblock1.denselayer6.conv2',
                      'features.conv0']
        elif args.model == 'ResNet101':
            layers = ['conv1', 'layer1.2.conv3.weight', 'layer2.3.conv3.weight', 'layer3.22.conv3.weight',
                      'layer4.2.conv3.weight', 'fc.weight']
        elif args.model == 'VGG19':
            layers = ['classifier.6.weight', 'features.50.weight', 'features.40.weight', 'features.30.weight',
                      'features.20.weight', 'features.10.weight', 'features.0.weight']

    for i in range(0, args.nmodels):

        for layer in layers:
            # Defining models
            if args.model == 'DenseNet161':
                model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 2)
            elif args.model == 'ResNet101':
                model = models.resnet101(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
            elif args.model == 'VGG19':
                model = models.vgg19_bn(pretrained=True)
                model.classifier.add_module('6', nn.Linear(4096, 2))

            # Loading weights
            weights = torch.load(args.modelPath)
            model.load_state_dict(weights['state_dict'])

            if layer is not None:
                resultPath = args.resultPath + layer.replace('.', '_') + '/'
                if not os.path.exists(resultPath):
                    os.makedirs(resultPath)

            modelList = []
            relChange = []
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

            # Loading the dataset
            livDetIris20_ds = LivDetIris2020(imageFolder=args.imageFolder, splitPath=args.splitPath)
            livDetIris20_dl = DataLoader(livDetIris20_ds, batch_size=16, shuffle=False,
                                         num_workers=int(os.cpu_count() * 0.5))

            # Generating predictions on the model
            for model_idx, model in enumerate(modelList):
                results = []
                for batch_idx, datasets in enumerate(
                        tqdm(livDetIris20_dl, desc=f"Running with scale: {scales[model_idx]:0.3f} ")):
                    data, imgName, label = datasets
                    if model_idx == 0:
                        testImgNames.extend(imgName)
                        testTrueLabels.extend(list(label.numpy()))
                    data = data.to(device)
                    predictions = model(data).detach().cpu().numpy()[:, 1]
                    results.extend(predictions)
                testPredScores[model_idx] = np.array(results)

            print(testPredScores.shape, "final predictions")
            print(testPredScores[0].min(), testPredScores[0].max(), "final predictions")

            with open(resultPath + args.perturbation + '.pickle', 'wb') as f:
                pickle.dump([testImgNames, testPredScores, testTrueLabels], f)

            for index, predict_score in enumerate(testPredScores):
                print(type(predict_score))
                # Normalization of scores in [0,1] range
                predictScore = (predict_score - min(predict_score)) / (max(predict_score) - min(predict_score))
                print(predictScore[:10], testTrueLabels[:10])
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
                        best_model = modelList[index].to("cpu")
                        torch.save(best_model.state_dict(), resultPath + args.perturbation + f"-{TDR:0.4f}-.pth")

                with open(resultPath + 'TDR-' + args.perturbation + '.csv', mode='a+') as fout:
                    fout.write("%f,%f,%f\n" % (scales[index], TDR, relChange[index]))
                print("TDR @ 0.002 FDR with %s : %f @ %f \n" % (args.perturbation, TDR, scales[index]))

import argparse
import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from numpy import linalg
from sklearn.metrics import roc_curve
from tqdm import tqdm

from config import base_model_paths, bestTDRs, get_layers
from dataset import get_LivDetIris2020
from fusions import loadNewModel
from perturbation import (weightPertubationDenseNet161,
                          weightPertubationResNet101,
                          weightPertubationVGG19)
from utils import get_cuda_device, make_search_index, update_models_details

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-perturbation', default='GaussianNoise', type=str,
                        help='GaussianNoise, WeightsZero, WeightsScaling, TopWeightsZero, BottomWeightsZero,'
                             'WeightsZeroScaling, Quantize, FiltersZero')
    parser.add_argument('-perturbationSetup', default='Layers', type=str, help='Entire, Layers')
    parser.add_argument('-scales', default='0.1', type=str, help='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('-splitPath', default='Data-Splits/LivDet-Iris-2020/test_split-Seg.csv', type=str)
    parser.add_argument('-imageFolder', default='Iris_Image_Database/', type=str)
    parser.add_argument('-modelPath', default='Model/LivDet-Iris-2020/DenseNet161_best.pth', type=str)
    parser.add_argument('-resultPath', default='Results/LivDet-Iris-2020/', type=str)
    parser.add_argument('-model', default='DenseNet161', type=str, help='DenseNet161, ResNet101, VGG19')
    parser.add_argument('-bestTDR', default=0.9022)
    parser.add_argument('-nmodels', default=1, type=int)
    parser.add_argument("-device", default="cuda:0", type=str)
    args = parser.parse_args()

    # CUDA Device assignment.
    args.device = get_cuda_device()

    # Selecting the best TDR for the model and setting the base model path
    args.bestTDR = bestTDRs[args.model]
    args.modelPath = base_model_paths[args.model]
    print(f"\n\nExperiment: \nModel: {args.model}\nModelPath: {args.modelPath}")
    print(f"Dataset: {args.splitPath} \nBestTDR: {args.bestTDR} \nDevice: {args.device}\n\n")

    # Creating result directory
    resultPath = os.path.join(args.resultPath, args.perturbation, args.model)
    os.makedirs(resultPath, exist_ok=True)

    # Defining scales
    scales = [float(scale) for scale in args.scales.split(',')]

    # Get the layers to be perturbed
    layers = get_layers(model=args.model, perturbationSetup=args.perturbationSetup)

    for i in range(0, args.nmodels):
        print(f"\n\n\nModel {i + 1} of {args.nmodels}")
        for layer in layers:
            # Loading the model
            model = loadNewModel(args.model, savedPath=args.modelPath, device=args.device)

            if layer is not None:
                resultPath = args.resultPath + layer.replace('.', '_') + '/'
                os.makedirs(resultPath, exist_ok=True)

            modelList, relChange = [], []
            for scale in scales:
                modelTemp = copy.deepcopy(model)

                # Perturbing models
                print(f"Layer {layer}")
                if args.model == 'DenseNet161':
                    modelTemp = weightPertubationDenseNet161(modelTemp, layer, args.perturbation, scale).to(args.device)
                elif args.model == 'ResNet101':
                    modelTemp = weightPertubationResNet101(modelTemp, layer, args.perturbation, scale).to(args.device)
                elif args.model == 'VGG19':
                    modelTemp = weightPertubationVGG19(modelTemp, layer, args.perturbation, scale).to(args.device)

                # Calculating overall relative difference in the parameters
                diffParameters = torch.cat([(param_1 - param_2).view(-1) for param_1, param_2 in
                                            zip(modelTemp.parameters(), model.parameters())], dim=0)
                orgParameters = torch.cat([param_2.view(-1) for param_2 in model.parameters()], dim=0)
                relChange.append(
                    linalg.norm(diffParameters.detach().cpu().numpy()) / linalg.norm(
                        orgParameters.detach().cpu().numpy()))
                modelTemp = modelTemp.to(args.device)
                modelTemp.eval()
                modelList.append(modelTemp)

            testData = pd.read_csv(args.splitPath, header=None)
            print("Number of models", len(modelList), "for scale(s):", scales)
            testPredScores = np.zeros((len(modelList), len(testData.values)))
            testTrueLabels, testImgNames, segInfo = [], [], None

            # Loading the dataset
            livDetIris20_ds, livDetIris20_dl = get_LivDetIris2020(imageFolder=args.imageFolder,
                                                                  splitPath=args.splitPath)

            # Generating predictions on the model
            for model_idx, model in enumerate(modelList):
                results = []
                for batch_idx, datasets in enumerate(
                        tqdm(livDetIris20_dl, desc=f"Running with scale: {scales[model_idx]:0.3f} ")):
                    data, imgName, label = datasets
                    if model_idx == 0:
                        testImgNames.extend(imgName)
                        testTrueLabels.extend(list(label.numpy()))
                    data = data.to(args.device)
                    predictions = model(data).detach().cpu().numpy()[:, 1]
                    results.extend(predictions)
                testPredScores[model_idx] = np.array(results)

            # print(testPredScores.shape, "final predictions")
            # print(testPredScores[0].min(), testPredScores[0].max(), "final predictions")

            with open(resultPath + args.perturbation + '.pickle', 'wb') as f:
                pickle.dump([testImgNames, testPredScores, testTrueLabels], f)

            for index, predict_score in enumerate(testPredScores):

                # Normalization of scores in [0,1] range
                predictScore = (predict_score - min(predict_score)) / (max(predict_score) - min(predict_score))

                (fprs, tprs, thresholds) = roc_curve(testTrueLabels, predictScore)
                imgNameScore = []
                for k in range(len(testImgNames)):
                    imgNameScore.append([testImgNames[k], testTrueLabels[k], predictScore[k]])

                # Calculating TDR at 0.2% FDR
                TDR = np.interp(0.002, fprs, tprs)

                model_save_dir = os.path.join(args.resultPath, args.perturbation, args.model)
                os.makedirs(model_save_dir, exist_ok=True)

                # Storing results
                if TDR > args.bestTDR:
                    # Save the best model
                    best_model = modelList[index].to("cpu")
                    torch.save(best_model.state_dict(),
                               os.path.join(model_save_dir,
                                            f"{args.model}{args.perturbation}-{TDR:0.4f}-.pth"))

                    # Update models_detail.csv and keep the best 30 models
                    update_models_details(filePath=os.path.join(model_save_dir, f"models_detail.csv"),
                                          keep_best=20,
                                          info={"fileName": f"{args.model}{args.perturbation}-{TDR:0.4f}-.pth",
                                                "layer": layer,
                                                "scale": scales[index],
                                                "TDR": round(TDR, 4),
                                                "relChange": round(relChange[index], 6)})

                    print(f"Improvement Found: ", f"{args.model}{args.perturbation}-{TDR:0.4f}-.pth above",
                          args.bestTDR)

                else:
                    print(f"No Improvement Found: ", f"{args.model}{args.perturbation}-{TDR:0.4f}-.pth above",
                          args.bestTDR)

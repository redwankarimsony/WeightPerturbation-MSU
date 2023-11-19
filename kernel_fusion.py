
import argparse
import glob
import os
import time

import torch

from config import bestTDRs, base_model_paths, get_layers
from dataset import get_LivDetIris2020
from fusions import (kernelFusionDenseNet161,
                     kernelFusionVGG19,
                     kernelFusionResNet101,
                     loadNewModel,
                     loadSummary,
                     getOriginalLabels,
                     getPrediction,
                     getScore)
from utils import get_cuda_device

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-perturbation', default='GaussianNoise', type=str,
                        help='GaussianNoise, WeightsZero, WeightsScaling, TopWeightsZero, '
                             'BottomWeightsZero, WeightsZeroScaling, Quantize, FiltersZero')
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

    # Selecting the best TDR for the model and setting the base model path
    args.bestTDR = bestTDRs[args.model]
    args.modelPath = base_model_paths[args.model]
    print(f"Experiment: \nModel: {args.model}\nModelPath: {args.modelPath}")
    print(f"\nDataset: {args.splitPath} BestTDR: {args.bestTDR}\nDevice: {args.device}")

    # CUDA Device assignment.
    device = get_cuda_device()

    # Creating result directory
    resultPath = os.path.join(args.resultPath, args.perturbation, args.model)
    os.makedirs(resultPath, exist_ok=True)

    # Defining scales
    scales = [float(scale) for scale in args.scales.split(',')]


    layers = get_layers(model=args.model, perturbationSetup=args.perturbationSetup)

    # Loading the dataset
    livDetIris20_ds, livDetIris20_dl = get_LivDetIris2020(imageFolder=args.imageFolder,
                                                          splitPath=args.splitPath)

    # Listing all the saved weights files
    saved_weights = glob.glob(os.path.join(args.resultPath, args.perturbation, "*.pth"))
    # print(saved_weights)

    combinations = loadSummary(os.path.join(args.resultPath, args.perturbation, "summary.csv"))
    print(combinations.head())

    for rowIdx, row in combinations.iterrows():
        if combinations.iloc[rowIdx, 4] == "None" or combinations.iloc[rowIdx, 4] == "None":
            print("\n\n Row ID: " + rowIdx, "\n", row)
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
            if args.model == 'DenseNet161':
                model_Kfused = kernelFusionDenseNet161(modelA, modelB, layer=None)
            elif args.model == 'ResNet101':
                model_Kfused = kernelFusionResNet101(modelA, modelB, layer=None)
            elif args.model == 'VGG19':
                model_Kfused = kernelFusionVGG19(modelA, modelB, layer=None)
            else:
                raise ValueError("Model not supported")
            model_Kfused.eval()

            # Generate predictions on model pair
            modelA_preds = getPrediction(model=modelA, dataLoader=livDetIris20_dl, device=torch.device("cuda:1"))
            modelB_preds = getPrediction(model=modelB, dataLoader=livDetIris20_dl, device=torch.device("cuda:0"))

            # Remove models from memory
            modelB, modelA = None, None
            torch.cuda.empty_cache()

            # Generate predictions on kernel fused model
            model_Kfused_preds = getPrediction(model=model_Kfused, dataLoader=livDetIris20_dl,
                                               device=torch.device("cuda:1"))
            model_Kfused = None
            torch.cuda.empty_cache()

            time.sleep(3)
            print(modelA_preds.shape, modelB_preds.shape, model_Kfused_preds.shape)
            # Perform Score Fusion
            regFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores=(modelA_preds + modelB_preds) / 2,
                                      threshold=0.002)

            # Perform Kernel Fusion
            kernelFusionScore = getScore(testTrueLabels=testTrueLabels, rawScores=model_Kfused_preds, threshold=0.002)
            print(f"reg fusion: {regFusionScore}, kernel fusion: {kernelFusionScore}")

            combinations.iloc[rowIdx, 4] = regFusionScore
            combinations.iloc[rowIdx, 5] = kernelFusionScore

            combinations.to_csv(os.path.join(args.resultPath, args.perturbation, "summary.csv"), index=False)

        else:
            print(f"Row number {rowIdx + 1} already calculated")

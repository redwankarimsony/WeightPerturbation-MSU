import torch
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
from data_conversion import *
import random
import torchvision.datasets as dset
from models.utils import AverageMeter
import models
import torchvision
import matplotlib.pyplot as plt
from Evaluation import evaluation


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
        tranform_img=[]
        if len(SegInfo) >= 3:
            min_x = math.floor(int(SegInfo[0]) - int(SegInfo[2]) - 5)
            min_y = math.floor(int(SegInfo[1]) - int(SegInfo[2]) - 5)
            max_x = math.floor(int(SegInfo[0]) + int(SegInfo[2]) + 5)
            max_y = math.floor(int(SegInfo[1]) + int(SegInfo[2]) + 5)
            image = image.crop([min_x, min_y, max_x, max_y])

            tranform_img = valTransform(image)
            tranform_img = tranform_img.repeat(3, 1, 1)
            isImageSeg=True
    except:
        isImageSeg= False
    return tranform_img,isImageSeg

def validate(val_loader, model,device):
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        true_labels= []
        pred_labels =[]
        for i, (input, target) in enumerate(val_loader):

            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prediction, prec = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec[0].item(), input.size(0))
            true_labels.extend(target.tolist())
            pred_labels.extend(prediction.tolist())

        # obvResult = evaluation()
        # obvResult.get_result_multiclass('ResNet44', list(np.array(true_labels).flat), list(np.array(pred_labels).flat), '../TempData/CIFAR10_Logs_Results/')

    return top1.avg

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        _, prediction = output.topk(1, 1, True, True)
        return prediction, res

def weightPertubation(model, layer, perturbation, proportion):

    # weights = model.classifier.weight.detach().numpy()
    # gaussian_scale = proportion * np.std(weights)
    # weights =weights+ np.random.normal(loc=0.0, scale=gaussian_scale, size=weights.shape)
    # model.classifier.weight = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))
    #
    # bias = model.classifier.bias.detach().numpy()
    # gaussian_scale = proportion * np.std(bias)
    # bias = bias + np.random.normal(loc=0.0, scale=gaussian_scale, size=bias.shape)
    # model.classifier.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    # Overall change of the the weight architecture
    #VGG19
    # layers = ['0','3','7','10','14','17','20','23','27','30','33','36','40','43','46','49']
    #layer is None and (any(number in name for number in layers) or 'classifier' in name)
    # For ResNet101
    # layer is None and ('conv' in name or 'fc' in name)
    # For DenseNet121
    # layer is None and ('conv' in name or 'classifier' in name)

    for name, param in model.named_parameters():
        #print(name)
        if (layer is None and ('conv' in name or 'classifier' in name)) or (layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().cpu().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights), size=weights.shape)
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
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                elif perturbation =='BitFlip':
                    flatten_weight = weights.flatten()
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    bin_w = int2bin(flatten_weight[indices], 8)#.short()
                    bit_idx = random.choice(range(8))
                    mask = (bin_w.clone().zero_() + 1) * (2 ** bit_idx)
                    bin_w = bin_w ^ mask
                    int_w = bin2int(bin_w, 8).float()
                    flatten_weight[indices] = int_w
                    weights = flatten_weight.view(weights.shape)
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
                elif perturbation =='BottomWeightsZero':
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

if __name__ == '__main__':

    #plotTDRvsPurtabations()
    parser = argparse.ArgumentParser()
    parser.add_argument('-perturbation', default='BottomWeightsZero', type=str, help= 'GaussianNoise,WeightsZero, WeightsScaling, TopWeightsZero, BottomWeightsZero, WeightsZeroScaling, Quantize')
    parser.add_argument('-data_path', default='../TempData/CIFAR10_Data/', type=str, help='Path to dataset')
    parser.add_argument('-modelPath', default='../TempData/CIFAR10_Logs_Results/DenseNet121/model_best.pth.tar', type=str)
    parser.add_argument('-resultPath', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/BottomWeightsZero/CIFAR10_Layers/DenseNet121/', type=str)
    parser.add_argument('--test_batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('-model', default='DenseNet121', type=str, help='DenseNet121,ResNet44')
    parser.add_argument('-quantize', default='False', type=bool)
    args = parser.parse_args()
    # os.environ['TORCH_HOME'] = '../TempData/models/'

    device = torch.device('cuda')

    # Data Description
    if args.model == 'ResNet44':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        bestAccuracy = 83.77  # 96.75, 83.77

    else:
        test_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        bestAccuracy = 96.75  # 96.75, 83.77

    test_data = dset.CIFAR10(args.data_path,
                             train=False,
                             transform=test_transform,
                             download=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    # Perturbation Parameters
    scales=  [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    nModels = 1

    for i in range(0, nModels):
        model = models.__dict__['vanilla_resnet44'](10) if args.model == 'ResNet44' else torchvision.models.densenet121(pretrained=True)
        layers = ['features.denseblock4.denselayer24.conv2'] #[name for name, param in model.named_parameters() if 'conv' in name] #, [None]
        bestScale = 0.0
        bestLayer = None
        for layer in layers:
            if args.model == 'ResNet44':
                model = models.__dict__['vanilla_resnet44'](10)
            else:
                model = torchvision.models.densenet121(pretrained=True)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 10)

            weights = torch.load(args.modelPath)
            model.load_state_dict(weights['state_dict'])

            if layer is not None:
                layerResultPath = args.resultPath + layer.replace('.', '_') + '/'
            else:
                layerResultPath = args.resultPath
            if not os.path.exists(layerResultPath):
                os.makedirs(layerResultPath)

            modelList = []
            relChange = []
            for scale in scales:
                modelTemp = copy.deepcopy(model)
                modelTemp = weightPertubation(modelTemp, layer, args.perturbation, scale)
                diffParameters = torch.cat([(param_1 - param_2).view(-1) for param_1, param_2 in zip(modelTemp.parameters(), model.parameters())],dim=0)
                orgParameters = torch.cat([param_2.view(-1) for param_2 in model.parameters()], dim=0)
                relChange = linalg.norm(diffParameters.detach().numpy()) / linalg.norm(orgParameters.detach().numpy())
                modelTemp = modelTemp.to(device)
                modelTemp.eval()
                modelList.append(modelTemp)

                val_acc = validate(test_loader, modelTemp,device)

                if val_acc >= bestAccuracy:
                    with open(args.resultPath + 'ImprovedTDRs-' + args.model+ '-'+ args.perturbation + '.csv', mode='a+') as fout:
                        fout.write("%s,%f,%f,%f\n" % (layer,scale, val_acc, relChange))
                with open(layerResultPath + 'TDR-'+args.model+ '-'+ args.perturbation+'.csv', mode='a+') as fout:
                    fout.write("%f,%f,%f\n" % (scale, val_acc, relChange))
                print("Accuracy with %s : %f @ %f \n" % (args.perturbation, val_acc, scale))

    #obvResult = evaluation()
    #errorIndex, predictScore, threshold = obvResult.get_result(args.model, testImgNames, testTrueLabels, testPredScores,args.resultPath)

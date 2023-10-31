import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import math
import numpy as np
import torch.nn.functional as F
import argparse
from Evaluation import evaluation
from PIL import Image, ImageOps
import pickle
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import svm
from sklearn.decomposition import PCA
import pandas as pd
from pytorch_pretrained_vit.model import AnomalyViT, ViT
from numpy.linalg import eig
from sklearn import mixture
from sklearn.covariance import GraphicalLasso
from sklearn.neighbors import LocalOutlierFactor


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViT('B_16', pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        # self.backbone = models.densenet201(pretrained=True)
        # self.backbone.classifier = torch.nn.Identity()
    def forward(self, x):

        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

def distance_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    distance = pairwise_distances(test_set, train_set, metric="cosine")
    distance.sort(axis=1)
    D = [i[0:n_neighbours] for i in distance]
    return np.mean(D, axis=1)

def GMM_Model(train_features):
    train_features = np.array(train_features)
    cov_train_features = np.cov(train_features.T)
    values, vectors = eig(cov_train_features)
    sorted_vals = sorted(values, reverse=True)
    cumsum_vals = np.cumsum(sorted_vals)
    explained_vars = cumsum_vals / cumsum_vals[-1]

    for i, explained_var in enumerate(explained_vars):
        n_components = i
        if explained_var > 0.9:
            break

    pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
    train_features = np.ascontiguousarray(pca.fit_transform(train_features))

    # build GMM
    dens_model = mixture.GaussianMixture(n_components=1,
                                         max_iter=1000,
                                         verbose=1,
                                         n_init=1)
    dens_model.fit(train_features)
    return pca, dens_model

valTransform = transforms.Compose([
                transforms.Resize([224, 224]), #384, 384   224, 224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

def preprocess_image(imgFile, SegInfo):
    image = Image.open(imgFile)
    imageName= imgFile.split('/')[-1]

    # Segmentation and cropping of an image
    isImageSeg=False
    if len(SegInfo) == 3:
        min_x = math.floor(int(SegInfo[0]) - int(SegInfo[2]) - 5)
        min_y = math.floor(int(SegInfo[1]) - int(SegInfo[2]) - 5)
        max_x = math.floor(int(SegInfo[0]) + int(SegInfo[2]) + 5)
        max_y = math.floor(int(SegInfo[1]) + int(SegInfo[2]) + 5)
        image = image.crop([min_x, min_y, max_x, max_y])
        # image = ImageOps.mirror(image)
        isImageSeg = True
        #image.save('../TempData/Iris_LivDet_2020_Results/GCT4/SegmentedImages/VeriEye/' + imageName)

    tranform_img = valTransform(image)
    tranform_img = tranform_img.repeat(3, 1, 1).unsqueeze(0)

    return tranform_img,isImageSeg

def weightDistribution():

    with open(args.resultPath + 'weights.pickle', "rb") as f:
        weightss,PAScores,testTrueLabels = pickle.load(f)


    minWeight = 0.4553  # 0.4585
    maxWeight = 0.5447   # 0.5415
    oldWeights = [(weightss[i][1]-minWeight)/(maxWeight-minWeight) for i in range(len(weightss))]
    newWeights = [(weightss[j][0]-minWeight)/(maxWeight-minWeight) for j in range(len(weightss))]
    testPredScores=[oldWeights[i]*PAScore[0]+ newWeights[i]*PAScore[1] for i, PAScore in enumerate(PAScores)]

    bins = np.linspace(np.min(np.array(oldWeights + newWeights)), np.max(np.array(oldWeights + newWeights)), 60)
    plt.figure()
    plt.hist(oldWeights, bins, alpha=0.7, label='D-NetPAD_Old', density=True, edgecolor='black', facecolor='g')
    plt.hist(newWeights, bins, alpha=0.7, label='D-NetPAD_New', density=True, edgecolor='black', facecolor='r')
    plt.legend(loc='upper right', fontsize='medium')
    plt.xlabel('Weights', fontsize='medium')
    plt.ylabel('Frequency', fontsize='medium')
    plt.savefig(args.resultPath + "Weight-Distribution.png")

    obvResult = evaluation()
    errorIndex, predictScore, threshold = obvResult.get_result(args.method, 'None', testTrueLabels,
                                                               testPredScores, args.resultPath)

if __name__ == '__main__':


    # Testing for fusion
    parser = argparse.ArgumentParser()

    # LivDet-2017 Experiments
    parser.add_argument('-csvFile', default='../TempData/Iris_LivDet_2017_Splits/test_train_split07-Seg.csv', type=str)
    parser.add_argument('-resultPath', default='../TempData/Iris_LivDet_2017_Results/WeightPerturbations/D-NetPAD/Fusion/Split-07/',type=str)
    # parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Logs/GCT4/VGG19_best.pth', type=str)
    # parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Logs/GCT4/ResNet101_best.pth', type=str)
    # parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Logs/GCT4/DesNet161_best.pth', type=str)
    # parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/BottomWeightsZero/VGG19_Layers/features_20/VGG19_0.8.pth', type=str)
    # parser.add_argument('-modelPath2', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/GaussianNoise/VGG19_Entire/VGG19_0.3.pth', type=str)
    # parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/BottomWeightsZero/ResNet101_Layers/conv1/ResNet101_0.4.pth', type=str)
    # parser.add_argument('-modelPath2', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/GaussianNoise/ResNet101_Entire/ResNet101_0.1.pth', type=str)
    parser.add_argument('-modelPath2', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/BottomWeightsZero/D-NetPAD_Layers/features_denseblock1_denselayer6/D-NetPAD_0.9.pth',type=str)
    parser.add_argument('-modelPath', default='../TempData/Iris_LivDet_2020_Results/WeightPertubation/GaussianNoise/DenseNet161_Entire/D-NetPAD_0.1.pth', type=str)
    parser.add_argument('-imageFolder', default='G:/My Drive/Renu/Iris_Image_Database/', type=str,help='GCT3-Data/,LivDet-iris-2017/')
    parser.add_argument('-method', default='D-NetPAD', type=str, help='ResNet101, D-NetPAD, VGG19')
    parser.add_argument('-isFusion', default=True, type=bool)
    args = parser.parse_args()
    device = torch.device('cuda')

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['TORCH_HOME'] = '../TempData/models/'

    # weightDistribution()

    # Load weights of DesNet121 model
    #start = time.process_time()
    if args.method == 'ResNet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if args.isFusion == True:
            model2 = models.resnet101(pretrained=True)
            num_ftrs = model2.fc.in_features
            model2.fc = nn.Linear(num_ftrs, 2)
    elif args.method =='VGG19':
        model = models.vgg19_bn(pretrained=True)
        model.classifier.add_module('6', nn.Linear(4096, 2))
        if args.isFusion == True:
            model2 = models.vgg19_bn(pretrained=True)
            model2.classifier.add_module('6', nn.Linear(4096, 2))
    else:
        model = models.densenet161(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
        if args.isFusion == True:
            model2 = models.densenet161(pretrained=True)
            num_ftrs = model2.classifier.in_features
            model2.classifier = nn.Linear(num_ftrs, 2)

    modelWeights = torch.load(args.modelPath)
    model.load_state_dict(modelWeights['state_dict'])
    model = model.to(device)
    model.eval()

    if args.isFusion == True:
        modelWeights2 = torch.load(args.modelPath2)
        model2.load_state_dict(modelWeights2['state_dict'])
        model2 = model2.to(device)
        model2.eval()

    testPredScores =[]
    weightss=[]
    PAScores=[]
    testTrueLabels=[]
    testImgNames=[]
    successPAD = 0
    totalImages = 0
    SegInfo = []
    threshold = 0.5
    imagesScores = []
    count=0
    testData = pd.read_csv(args.csvFile, header=None)
    testData = testData.loc[(testData[0] == 'test')]
    for v in testData.values:
        if v[0] == 'test':
            count += 1
            print(count)
            imagePath = args.imageFolder + v[2]
            imageName = v[2].split('/')[-1]
            SegInfo = [v[-3], v[-2], v[-1]]   # for LivDet-2017

            # Segmentation of image
            tranformImage, isImageSeg = preprocess_image(imagePath, SegInfo)
            if isImageSeg:

                tranformImage = tranformImage.to(device)
                outputSB = model(tranformImage[:,0:3,:,:])
                PAScore1 = outputSB.detach().cpu().numpy()[:, 1]
                PAScore1 = np.minimum(np.maximum((PAScore1 + 15) / 35, 0), 1)
                PAScore = PAScore1[0]

                if args.isFusion == True:
                    outputSB2 = model2(tranformImage[:, 0:3, :, :])
                    PAScore2 = outputSB2.detach().cpu().numpy()[:, 1]
                    PAScore2 = np.minimum(np.maximum((PAScore2 + 15) / 35, 0), 1)
                    PAScore = PAScore1[0]+ PAScore2[0]

                testPredScores.append(PAScore)
                testImgNames.append(v[2])
                # testTrueLabels.append(0)
                if  v[1] == 'Live':
                #if '/T1/' in imagePath:
                    testTrueLabels.append(0)
                else:
                    testTrueLabels.append(1)

    obvResult = evaluation()
    errorIndex, predictScore, threshold = obvResult.get_result(args.method, testImgNames, testTrueLabels, testPredScores,args.resultPath)

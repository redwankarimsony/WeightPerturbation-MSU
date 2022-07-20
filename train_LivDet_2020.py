import os
import sys
import argparse
import inspect
import datetime
import json
from Evaluation import evaluation
import time
import matplotlib.pyplot as plt
import numpy as np
from IARPA_dataset import IARPA_dataset
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-num_epochs', type=int, default=50)
parser.add_argument('-splits', default= 'Split-01',type=str)
parser.add_argument('-csv_path', default= '../TempData/Iris_LivDet_2020_Splits/train_test_split_GCT4_PM-Seg-Edit.csv',type=str) #Iris_OCT_NIR_Splits
parser.add_argument('-database_path', default='G:/My Drive/Renu/Iris_Image_Database/',type=str)  #'G:/My Drive/Renu/IPARA-Project-Documentation/GCT2/OCT_Data/'
parser.add_argument('-result_path', default= '../TempData/Iris_LivDet_2020_',type=str)  # '../TempData/Iris_OCT_NIR_'
#parser.add_argument('-modelPath', default='../TempData/Iris_IARPA_Logs/GCT3/DesNet121_save_best.pth',type=str)
parser.add_argument('-method', default= 'InceptionV3',type=str)  # DenseNet121, VGG19, ResNet50

#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#os.environ['TORCH_HOME'] = '../TempData/models/'

args = parser.parse_args()
device = torch.device('cuda')

if args.method =='VGG19':
    model = models.vgg19_bn(pretrained=True)
    model.classifier.add_module('6', nn.Linear(4096,2))
elif args.method =='ResNet50':
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif args.method =='DenseNet121':
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
elif args.method =='DenseNet161':
    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
elif args.method =='DenseNet201':
    model = models.densenet201(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
elif args.method =='InceptionV3':
    model = models.inception_v3(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)


#model = nn.DataParallel(model).to(device)
model = model.to(device)

log_path = os.path.join(args.result_path + 'Logs/GCT4/' + args.method +'/')
if not os.path.exists(log_path):
    os.makedirs(log_path)
result_path = os.path.join(args.result_path + 'Results/GCT4/' + args.method +'/')
if not os.path.exists(result_path):
    os.makedirs(result_path)
method = args.method

## For LivDet-2017 Dataset
dataseta = IARPA_dataset(args.csv_path,args.database_path,train_test='train')
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

dataset = IARPA_dataset(args.csv_path,args.database_path, train_test='test', c2i=dataseta.class_to_id)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

dataloader = {'train': dl, 'test':test}


lr = 0.005
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

# deal with hyper-params...
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}


###############
#
# Train the model and save everything
#
###############
obvResult = evaluation()
num_epochs =args.num_epochs
train_loss=[]
val_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
for epoch in range(num_epochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        e=s=0
        valPredScore = []
        valTrueLabel = []
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(train):
            for data, cls, imageName in dataloader[phase]:
                data = data.to(device)
                cls = cls.to(device)
                
                outputs = model(data)
                if args.method == 'InceptionV3':
                    outputs = outputs.logits
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                loss = F.cross_entropy(outputs, cls)
                #print(loss)
                
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'val':
                    valPredScore.extend(outputs.detach().cpu().numpy())
                    valTrueLabel.extend(cls.detach().cpu().numpy())
                elif phase == 'test':
                    testPredScore.extend(outputs.detach().cpu().numpy())
                    testTrueLabel.extend(cls.detach().cpu().numpy())
                    imgNames.extend(imageName)

                    
                tloss += loss.item()
                c += 1
            
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)
        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            lr_sched.step(tloss / c)
            val_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy = accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path, method + '_best.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames = imgNames


    with open(os.path.join(log_path,method+'_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(model.state_dict(), os.path.join(log_path, method+'_model.pt'))

plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, num_epochs), train_loss[:], color='r')
plt.plot(np.arange(0, num_epochs), val_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(result_path + method+'_Loss.jpg')

obvResult.get_result(args.method, testImgNames, testTrueLabels, testPredScores, result_path)

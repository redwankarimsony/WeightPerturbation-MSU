import os
import time
import math
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet161
from torchvision.models.densenet import DenseNet161_Weights
from tqdm import tqdm

valTransform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])


class LivDetIris2020(Dataset):
    def __init__(self, imageFolder, splitPath) -> None:
        """
        This is the LivDet-Iris-2020 and it is used to load the data
        for being used in pytorch models.
        """
        super().__init__()
        self.imageFolder = imageFolder
        self.splitPath = splitPath
        self.testData = pd.read_csv(self.splitPath, header=None)

    def __len__(self):
        return len(self.testData)

    def preprocess_image(self, imgFile, SegInfo):
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
            print("Something wrong")
        return tranform_img, isImageSeg

    def __getitem__(self, index):
        v = self.testData.values[index]
        # print(v)
        imageName = v[2]
        imagePath = os.path.join(self.imageFolder ,  imageName)
        segInfo = v[3:]

        # Segmentation of image
        tranformImage, isImageSeg = self.preprocess_image(imagePath, segInfo)
        return tranformImage, imageName, 0 if v[1] =="Live" else 1
        
    

if __name__ == "__main__":
    ds = LivDetIris2020(imageFolder="Iris_Image_Database/", 
                        splitPath="Data-Splits/LivDet-Iris-2020/test_split-Seg.csv")

    data, isImageSeg, label = ds[0]
    print(data.shape, isImageSeg, label)

    ncpus = int(os.cpu_count()*0.5)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers= ncpus)

    # Benchmarking data loading time
    t = time.time()
    # for batch_idx, datasets  in enumerate(dl):
    #     data, seg, label = datasets
    #     # print(batch_idx)
    # #     print(data.shape)
    # print(f"Data Loading Time with {ncpus} cpus: {time.time()-t: 0.2f} sec.")


    model = densenet161(weights=DenseNet161_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 2)

    # Loading weights
    weights = torch.load('Model/LivDet-Iris-2020/DesNet161_best.pth')
    model.load_state_dict(weights['state_dict'])

        # CUDA Device assignment.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.eval()
    model.to(device)
    
    results = []
    for batch_idx, datasets  in enumerate(tqdm(dl)):
        data, imgName, label = datasets
        
        data = data.to(device)
        predictions = model(data).detach().cpu()[:, 1]
        results.append(predictions)
    
    results = torch.cat(results)
    print(results.shape)


    

#!/bin/bash

for VARIABLE in {1..100}
do
    python perturb_model-ResNet101.py -modelPath 'Model/LivDet-Iris-2020/ResNet101_best.pth' -model 'ResNet101' -device "cuda:0"
done
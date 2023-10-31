#!/bin/bash

for VARIABLE in {1..100}
do
    python perturb_model-ResNet101.py -modelPath 'Model/LivDet-Iris-2020/VGG19_best.pth' -model 'VGG19' -device "cuda:1" -scales "0.1"
done
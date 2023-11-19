#!/bin/bash


# Run perturbation search for 100 times for ResNet101
for VARIABLE in {1..100}
do
    python perturb_models.py -model 'ResNet101'
done

# Run perturbation search for 100 times for DenseNet161
for VARIABLE in {1..100}
do
    python perturb_models.py -model 'DenseNet161'
done


# Run perturbation search for 100 times for VGG19
for VARIABLE in {1..100}
do
    python perturb_models.py -model 'VGG19' -scales "0.1"
done






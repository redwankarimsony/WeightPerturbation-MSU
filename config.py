# List of best TDRs for each model

bestTDRs = {"DenseNet161": 0.9022,
            "ResNet101": 0.8411,
            "VGG19": 0.7687}

# List of base_model_paths
base_model_paths = {"DenseNet161": "Model/LivDet-Iris-2020/DesNet161_best.pth",
                    "ResNet101": "Model/LivDet-Iris-2020/ResNet101_best.pth",
                    "VGG19": "Model/LivDet-Iris-2020/VGG19_best.pth"}


def get_layers(model, perturbationSetup):
    # Defining perturbations selection layers based on the selected models
    if perturbationSetup == 'Entire':
        layers = [None]
    else:
        if model == 'DenseNet161':
            layers = ['features.denseblock4.denselayer24.conv2',
                      'features.denseblock3.denselayer36.conv2',
                      'features.denseblock2.denselayer12.conv2',
                      'features.denseblock1.denselayer6.conv2',
                      'features.conv0']
        elif model == 'ResNet101':
            layers = ['conv1',
                      'layer1.2.conv3.weight',
                      'layer2.3.conv3.weight',
                      'layer3.22.conv3.weight',
                      'layer4.2.conv3.weight',
                      'fc.weight']
        elif model == 'VGG19':
            layers = ['classifier.6.weight',
                      'features.50.weight',
                      'features.40.weight',
                      'features.30.weight',
                      'features.20.weight',
                      'features.10.weight',
                      'features.0.weight']

        else:
            raise ValueError("Model not supported")
    return layers


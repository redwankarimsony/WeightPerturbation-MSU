import numpy as np
import torch


def weightPertubationDenseNet161(model, layer, perturbation, proportion):
    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and ('conv' in name or 'classifier' in name)) or (layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().cpu().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
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
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
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
                elif perturbation == 'BottomWeightsZero':
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


def weightPertubationResNet101(model, layer, perturbation, proportion):
    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and ('conv' in name or 'fc' in name)) or (layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), -K)[-K:]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), K)[:K]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                param.data = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))
            elif 'bias' in name:
                bias = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    bias = bias + np.random.normal(loc=0.0, scale=proportion * np.std(bias), size=bias.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    bias = bias * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), -K)[-K:]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), K)[:K]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                    bias = bias * 5
                param.data = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    return model


def weightPertubationVGG19(model, layer, perturbation, proportion):
    layers = ['0', '3', '7', '10', '14', '17', '20', '23', '27', '30', '33', '36', '40', '43', '46', '49']

    for name, param in model.named_parameters():
        # print(name)
        if (layer is None and (any(number in name for number in layers) or 'classifier' in name)) or (
                layer is not None and layer in name):
            if 'weight' in name:
                weights = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    weights = weights + np.random.normal(loc=0.0, scale=proportion * np.std(weights),
                                                         size=weights.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    weights = weights * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), -K)[-K:]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(weights.size * proportion)
                    indices = np.argpartition(np.square(weights.flatten()), K)[:K]
                    weights[np.unravel_index(indices, weights.shape)] = 0
                elif perturbation == 'FiltersZero':
                    indices = np.random.choice(weights.shape[0], replace=False, size=int(weights.shape[0] * proportion))
                    if len(weights.shape) == 4:
                        weights[indices, :, :, :] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(weights.size, replace=False, size=int(weights.size * proportion))
                    weights[np.unravel_index(indices, weights.shape)] = 0
                    weights = weights * 5
                param.data = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float))
            elif 'bias' in name:
                bias = param.detach().numpy()
                if perturbation == 'GaussianNoise':
                    bias = bias + np.random.normal(loc=0.0, scale=proportion * np.std(bias), size=bias.shape)
                elif perturbation == 'WeightsZero':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsScaling':
                    bias = bias * proportion
                elif perturbation == 'TopWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), -K)[-K:]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'BottomWeightsZero':
                    K = int(bias.size * proportion)
                    indices = np.argpartition(np.square(bias.flatten()), K)[:K]
                    bias[np.unravel_index(indices, bias.shape)] = 0
                elif perturbation == 'WeightsZeroScaling':
                    indices = np.random.choice(bias.size, replace=False, size=int(bias.size * proportion))
                    bias[np.unravel_index(indices, bias.shape)] = 0
                    bias = bias * 5
                param.data = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float))

    return model

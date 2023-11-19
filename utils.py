import glob
import os
from itertools import combinations
import torch


def get_cuda_device():
    # CUDA Device assignment.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def make_search_index(result_dir, perturbation, model):
    search_str = os.path.join(result_dir, perturbation, f"{model}{perturbation}*.pth")

    files = glob.glob(search_str)

    # Generating all combinations of two files
    file_combinations = list(combinations(files, 2))

    # To see the combinations
    print("modelA,modelB,modelA-Score,modelB-Score")
    with open(os.path.join(result_dir, perturbation, f"summary-{model}.csv"), "w") as f:
        f.write("modelA,modelB,modelA-Score,modelB-Score\n")
        for (modelA, modelB) in file_combinations:
            f.write(
                modelA.split("/")[-1] + "," + modelB.split("/")[-1] + ',' + modelA.split("-")[-2] + ',' + modelB.split(
                    "-")[-2] + "\n")


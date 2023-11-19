import glob
import os
from itertools import combinations
import torch
import pandas as pd
import subprocess
import time
import os
import torch


def get_cuda_device():
    # CUDA Device assignment.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            best_gpu = get_gpu_with_least_memory_over_period()
            device = torch.device(f'cuda:{best_gpu}')
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
    with open(os.path.join(result_dir, perturbation, f"summary-{model}.csv"), "w") as f:
        f.write("modelA,modelB,modelA-Score,modelB-Score\n")
        for (modelA, modelB) in file_combinations:
            f.write(
                modelA.split("/")[-1] + "," + modelB.split("/")[-1] + ',' + modelA.split("-")[-2] + ',' + modelB.split(
                    "-")[-2] + "\n")
    
    print(f"Summary file saved at: {os.path.join(result_dir, perturbation, f'summary-{model}.csv')}")




def update_models_details(filePath, keep_best, info):
    model_save_path = os.path.dirname(filePath)
    
    models = [x for x in os.listdir(model_save_path) if x.endswith(".pth")]

    if not os.path.exists(filePath):
        # Write a dummy Dataframe
        column_names = ["fileName", "layer", "scale", "TDR", "relChange"]
        blank_df = pd.DataFrame(columns=column_names)

        # Add single row to the dataframe
        blank_df = blank_df.append(info, ignore_index=True)
        blank_df.to_csv(filePath, index=False)

    else:
        # Read the dataframe
        df = pd.read_csv(filePath)

        # Add single row to the dataframe
        df = df.append(info, ignore_index=True)

        # Sort the dataframe
        df = df.sort_values(by=['TDR'], ascending=False).reset_index(drop=True)

        # Drop duplicates
        df = df.drop_duplicates(subset=['fileName'], keep='first')

        # Keep best 30 models
        df = df.head(keep_best)

        # Save the dataframe
        df.to_csv(filePath, index=False)


        # Delete the models that are not in the dataframe
        for model in models:
            if model not in df.fileName.values:
                os.remove(os.path.join(model_save_path, model))
                print(f"Deleted model: {model}")
    


    
def get_gpu_usage():
    """Returns a list of tuples containing GPU id and its memory usage."""
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode('utf-8')
        gpu_usages = []
        for line in nvidia_smi_output.strip().split("\n"):
            index, memory = line.split(", ")
            gpu_usages.append((int(index), int(memory)))
        return gpu_usages
    except Exception as e:
        print(f"Error obtaining GPU information: {e}")
        return []

def get_gpu_with_least_memory_over_period(period=5, interval=1):
    """Returns the GPU index with the least average memory usage over a period."""
    samples = int(period / interval)
    accumulated_usages = {}

    for _ in range(samples):
        for index, memory in get_gpu_usage():
            if index not in accumulated_usages:
                accumulated_usages[index] = []
            accumulated_usages[index].append(memory)
        time.sleep(interval)

    avg_usages = {index: sum(usages) / len(usages) for index, usages in accumulated_usages.items()}
    if not avg_usages:
        return None
    best_gpu = min(avg_usages, key=avg_usages.get)
    print(f"Selected GPU: {best_gpu} (Average memory usage: {avg_usages[best_gpu]})")
    return best_gpu
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import MulticlassCalibrationError
from pathlib import Path
import os
import random

experiments_folder = Path("/fast/AG_Kainmueller/fabian/projects/hackathon_2024/experiments")
all_experiments = os.listdir(experiments_folder)

results = {"experiment_id": [], "calibration_error": []} 

for experiment_id in all_experiments: 
    file = experiments_folder.joinpath(experiment_id).joinpath("train/test_results/predictions.csv")
    try: 
        data = pd.read_csv(file)
    except: 
        continue
    
    true_labels = data["groundtruth"].to_numpy() # shift score to correspond to softmax index
    softmax_scores = data[data.columns[~data.columns.isin(['label', 'groundtruth','pred_class_name', 'pred_class', 'sample_name', 'Background', 'background'])]].to_numpy()

    metric = MulticlassCalibrationError(n_bins=10, norm='l1', num_classes=int(max(true_labels)))
    res = metric(torch.tensor(softmax_scores), torch.tensor(true_labels))

    results["experiment_id"].append(experiment_id)
    results["calibration_error"].append(res.item())

results_df = pd.DataFrame(results)
results_df.to_csv("calibration_results.csv", index=False)


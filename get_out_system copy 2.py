from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import torch
import torchextractor as tx
from lifelines.utils import concordance_index
from rich.progress import Progress, TextColumn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from dataloader import BraTSDataset


def rf():
    return RandomForestRegressor(random_state=42)


def svrp():
    return SVR(kernel="poly")


def svrr():
    return SVR()


def lr():
    return LinearRegression()


def knn():
    return KNeighborsRegressor()


module_path = Path(r"D:\vit_reg\Only_3d_4")

device = "cuda:0"

train_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "train")
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)
val_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "val")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
test_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "test")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

name = ["Optim", "TraditionModel", "MAE", "RMSE", "C-index", "SpearmanR"]

fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
#optims =(p for p in module_path.iterdir() if p.is_dir())
optims = [module_path / "RAdam"]
for num, optim in enumerate(optims):
    model = torch.load(optim / "model" / "best_epoch.pth")
    extract_model = tx.Extractor(model, ["output_model.2"]).to(device)

    x_train_feature = []
    y_train = []
    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age,
                resection) in enumerate(p.track(train_dataloader, task_id=task), 1):
            data, survival_day, age, resection = data.to(device), survival_day.to(
                device), age.to(device), resection.to(device)
            _, features = extract_model(data, age, resection)
            #result = model(data, age, resection)
            x_train_feature.append(
                np.hstack([
                    features["output_model.2"].detach().cpu().numpy().flatten(),
                    age.detach().cpu().numpy().flatten(),
                    resection.detach().cpu().numpy().flatten(),
                ]))
            y_train.append(survival_day.cpu().numpy()[0][0])

    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age,
                resection) in enumerate(p.track(val_dataloader, task_id=task), 1):
            data, survival_day, age, resection = data.to(device), survival_day.to(
                device), age.to(device), resection.to(device)
            _, features = extract_model(data, age, resection)
            #result = model(data, age, resection)
            x_train_feature.append(
                np.hstack([
                    features["output_model.2"].detach().cpu().numpy().flatten(),
                    age.detach().cpu().numpy().flatten(),
                    resection.detach().cpu().numpy().flatten(),
                ]))
            y_train.append(survival_day.cpu().numpy()[0][0])

    x_test_feature = []
    y_test = []
    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age,
                resection) in enumerate(p.track(test_dataloader, task_id=task), 1):
            data, survival_day, age, resection = data.to(device), survival_day.to(
                device), age.to(device), resection.to(device)
            _, features = extract_model(data, age, resection)
            #result = model(data, age, resection)
            x_test_feature.append(
                np.hstack([
                    features["output_model.2"].detach().cpu().numpy().flatten(),
                    age.detach().cpu().numpy().flatten(),
                    resection.detach().cpu().numpy().flatten(),
                ]))
            y_test.append(survival_day.cpu().numpy()[0][0])

    output = []
    nc = {
        "rf": "Random Forest",
        "svrp": "SVR(poly kernel)",
        "svrr": "SVR(rbf kernel)",
        "lr": "Linear Regression",
        "knn": "K Nearest Regression"
    }
    for num, i in enumerate([rf, svrp, svrr, lr, knn]):
        tmodel = i()
        tmodel.fit(x_train_feature, y_train)
        y_pred = tmodel.predict(x_test_feature)
        y_test_ret = test_dataset.scaler.inverse_transform(
            np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_ret = test_dataset.scaler.inverse_transform(
            np.array(y_pred).reshape(-1, 1)).flatten()
        output.append([
            optim.stem, i.__name__, y_test_ret, y_pred_ret,
            concordance_index(y_test_ret, y_pred_ret)
        ])
        #y_test_ret, y_pred_ret = output[-1][2:4]
        #output.sort(key=lambda x: x[4])
        #print(output[-1][1], output[-1][4])
        #y_test_ret, y_pred_ret = output[-1][2:4]
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        kmf = KaplanMeierFitter()
        kmf.fit(durations=y_test_ret, label="Truth")
        kmf.plot_survival_function(ax=axs[num % 5][num // 5], ci_show=True)
        kmf.fit(durations=y_pred_ret, label="Predict")
        kmf.plot_survival_function(ax=axs[num % 5][num // 5], ci_show=True)
        #pg.plot_blandaltman(y_test_ret, y_pred_ret, ax=axs[num % 5][num // 5])
        p_value = logrank_test(y_test_ret, y_pred_ret).p_value
        axs[num % 5][num // 5].set_title(
            f"{optim.stem} with {nc[i.__name__]} \n(c-index: {concordance_index(y_test_ret, y_pred_ret):.4f}, p_value: ({p_value:.4f})"
        )
plt.savefig("3-t.png")

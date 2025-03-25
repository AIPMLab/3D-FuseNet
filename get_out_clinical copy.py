import pandas as pd
import torch
import torchextractor as tx
from pathlib import Path
from rich.progress import Progress, TextColumn
from dataloader_kf import BraTSDataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pingouin as pg
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

import pingouin as pg
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 25})


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


plt.rcParams['font.family'] = ['Times New Roman']
module_path = Path(r"D:\vit_reg\Only_3d_4")

device = "cuda:0"
fold_ = 0

train_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "train",
    kfold=fold_,
    preprocess=False)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)
val_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "val",
    kfold=fold_,
    preprocess=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
test_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "test",
    preprocess=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

name = [
    "TraditionModel", "MAE", "RMSE", "C-index", "SpearmanR", "p-value", "hr", "95%-ci"
]

fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
for fold_ in range(5):
    with open(f"3_kfold_{fold_}.csv", "w", encoding="UTF-8") as f:
        f.write(",".join(name) + "\n")

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
                #result = model(data, age, resection)
                x_train_feature.append(
                    np.hstack([
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
                #result = model(data, age, resection)
                x_train_feature.append(
                    np.hstack([
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
                #result = model(data, age, resection)
                x_test_feature.append(
                    np.hstack([
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
        for i in [rf, svrp, svrr, lr, knn]:
            tmodel = i()
            tmodel.fit(x_train_feature, y_train)
            y_pred = tmodel.predict(x_test_feature)
            y_test_ret = np.array(y_test)
            y_pred_ret = y_pred
            """y_test_ret = test_dataset.scaler.inverse_transform(
                np.array(y_test).reshape(-1, 1)).flatten()
            y_pred_ret = test_dataset.scaler.inverse_transform(
                np.array(y_pred).reshape(-1, 1)).flatten()"""
            all_data = pd.DataFrame({
                "time":
                y_test_ret.tolist() + y_pred_ret.tolist(),
                "event": [1] * len(y_test_ret) + [1] * len(y_pred_ret),
                "type": [0] * len(y_test_ret) + [1] * len(y_pred_ret)
            })
            cph = CoxPHFitter()
            cph.fit(all_data, duration_col='time', event_col='event')
            variable = 'type'
            p_value_a = cph.summary.loc["type", 'p']
            p_value = logrank_test(y_test_ret, y_pred_ret).p_value
            print(p_value_a, p_value)
            print(p_value_a == p_value)
            hr = cph.summary.loc["type", 'exp(coef)']
            ci_lower = cph.summary.loc[variable, 'exp(coef) lower 95%']
            ci_upper = cph.summary.loc[variable, 'exp(coef) upper 95%']
            ns = [
                i.__name__,
                #y_test_ret, y_pred_ret,
                mean_absolute_error(y_test_ret, y_pred_ret),
                mean_squared_error(y_test_ret, y_pred_ret)**0.5,
                concordance_index(y_test_ret, y_pred_ret),
                spearmanr(y_test_ret, y_pred_ret)[0],
                p_value,
                hr,
                f"[{ci_lower:.4f},{ci_upper:.4f}]"
            ]
            f.write(",".join(map(str, ns)) + "\n")

            plt.figure(figsize=(10, 5))

            ax = pg.plot_blandaltman(y_test_ret, y_pred_ret)
            ax.set_xlabel("Mean of Survival Days")
            ax.set_ylabel("Difference of Survival Days")
            ax.set_xlim(0, 1400)
            ax.set_xticks(range(0, 1500, 100))
            ax.set_xticklabels(
                map(str, (0, "", 200, "", 400, "", 600, "", 800, "", 1000, "", 1200, "",
                          1400)))
            ax.set_ylim(-1000, 1500)
            ax.set_yticks(range(-1000, 1600, 250))
            ax.set_yticklabels(
                map(str, (-1000, "", -500, "", 0, "", 500, "", 1000, "", 1500)))
            ax.xaxis.label.set_fontsize(24)
            ax.yaxis.label.set_fontsize(24)
            plt.savefig(rf"Only_3d_4/{nc[i.__name__]}.pdf", bbox_inches='tight')

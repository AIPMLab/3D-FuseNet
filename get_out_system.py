import pandas as pd
import torch
import torchextractor as tx
from pathlib import Path
from rich.progress import Progress, TextColumn
from dataloader import BraTSDataset
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
from lifelines import KaplanMeierFitter

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


module_path = Path(r"D:\vit_reg\only_3d_4")
with_clinical = True
extract_line = "output_model.2"
#extract_line = "decoder_model.8"
device = "cuda:0"

train_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "train")
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)

val_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "val")
val_dataloader = DataLoader(val_dataset,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True)
test_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData", "test")
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)

name = ["Optim", "TraditionModel", "MAE", "RMSE", "C-index", "SpearmanR"]

#fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
for num, optim in enumerate(p for p in module_path.iterdir() if p.is_dir()):
    model = torch.load(optim / "model" / "best_epoch.pth")
    print(model)
    extract_model = tx.Extractor(model, [extract_line]).to(device)

    x_train_feature = []
    y_train = []
    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn(
                "Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age, resection) in enumerate(
                p.track(train_dataloader, task_id=task), 1):
            data, survival_day, age, resection = data.to(
                device), survival_day.to(device), age.to(device), resection.to(
                    device)
            if with_clinical:
                _, features = extract_model(data, age, resection)
            else:
                _, features = extract_model(data)

            #result = model(data, age, resection)
            x_train_feature.append(
                np.hstack([
                    features[extract_line].detach().cpu().numpy().flatten(
                    ),
                    #age.detach().cpu().numpy().flatten(),
                    #resection.detach().cpu().numpy().flatten(),
                ]))
            y_train.append(survival_day.cpu().numpy()[0][0])

    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn(
                "Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age, resection) in enumerate(
                p.track(val_dataloader, task_id=task), 1):
            data, survival_day, age, resection = data.to(
                device), survival_day.to(device), age.to(device), resection.to(
                    device)
            if with_clinical:
                _, features = extract_model(data, age, resection)
            else:
                _, features = extract_model(data)

            #result = model(data, age, resection)
            x_train_feature.append(
                np.hstack([
                    features[extract_line].detach().cpu().numpy().flatten(
                    ),
                    #age.detach().cpu().numpy().flatten(),
                    #resection.detach().cpu().numpy().flatten(),
                ]))
            y_train.append(survival_day.cpu().numpy()[0][0])

    x_test_feature = []
    y_test = []
    with torch.no_grad(), Progress(
            *Progress.get_default_columns(),
            TextColumn(
                "Loss={task.fields[loss]}, MSE={task.fields[mse]}")) as p:
        task = p.add_task("Testing", loss=0, mse=0)
        for i, (data, survival_day, age,
                resection) in enumerate(p.track(test_dataloader, task_id=task),
                                        1):
            data, survival_day, age, resection = data.to(
                device), survival_day.to(device), age.to(device), resection.to(
                    device)
            if with_clinical:
                _, features = extract_model(data, age, resection)
            else:
                _, features = extract_model(data)

            #result = model(data, age, resection)
            x_test_feature.append(
                np.hstack([
                    features[extract_line].detach().cpu().numpy().flatten(
                    ),
                    #age.detach().cpu().numpy().flatten(),
                    #resection.detach().cpu().numpy().flatten(),
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
        y_test_ret = test_dataset.scaler.inverse_transform(
            np.array(y_test).reshape(-1, 1)).flatten()
        y_pred_ret = test_dataset.scaler.inverse_transform(
            np.array(y_pred).reshape(-1, 1)).flatten()
        output.append([
            optim.stem, i.__name__,
            y_test_ret, y_pred_ret,
            concordance_index(y_test_ret, y_pred_ret)
        ])
    
        #output.sort(key=lambda x: x[4])
        #print(output[-1][1], output[-1][4])
        #y_test_ret, y_pred_ret = output[-1][2:4]
        #pg.plot_blandaltman(y_test_ret, y_pred_ret, ax=axs[num % 5][num // 5])
        #axs[num % 5][num // 5].set_title(f"{optim.stem} with {nc[output[-1][1]]}")
        plt.figure(figsize=(10, 5))
        #ax = pg.plot_blandaltman(y_test_ret, y_pred_ret)


        kmf = KaplanMeierFitter()
        kmf.fit(durations=y_test_ret, event_observed=[1 for i in y_test_ret], label="Truth")
        ax = kmf.plot_survival_function()
        kmf.fit(durations=y_pred_ret, event_observed=[1 for i in y_pred_ret], label="Predict")
        kmf.plot_survival_function(ax=ax)

        """ax.set_xlabel("Mean of Survival Days")
        ax.set_ylabel("Difference of Survival Days")
        ax.set_xlim(0, 1400)
        ax.set_xticks(range(0, 1500, 100))
        ax.set_xticklabels(
            map(str, (0, "", 200, "", 400, "", 600, "", 800, "", 1000, "", 1200, "",
                        1400)))
        ax.set_ylim(-1000, 1500)
        ax.set_yticks(range(-1000, 1600, 250))
        ax.set_yticklabels(
            map(str, (-1000, "", -500, "", 0, "", 500, "", 1000, "", 1500)))"""
        ax.xaxis.label.set_fontsize(24)
        ax.yaxis.label.set_fontsize(24)
        plt.savefig(module_path / rf"{optim.stem}_and_{nc[output[-1][1]]}.pdf", bbox_inches='tight')
        pd.DataFrame({"y_true": y_test_ret, "y_pred": y_pred_ret}).to_csv(module_path / rf"{optim.stem}_and_{nc[output[-1][1]]}.csv")
#plt.savefig("2-t.png")
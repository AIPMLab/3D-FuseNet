import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from rich.progress import track
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from dataloader import BraTSDataset
from image_encoder3D import VitRegressionModel

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 25})

device = "cuda:0"

dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "test",
    using_seg=False,
    using_mri=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#epoch = int(sys.argv[1]) if sys.argv[1:] else 99
#model = torch.load(rf"model/epoch_{epoch}.pth").cuda()
#print(type(model))

criterion = nn.L1Loss().cuda()
val_criterion = nn.MSELoss().cuda()


def val(ax=None):
    # 1. 训练数据集
    loader = dataloader
    # 2. 进入训练模式
    model.eval()
    #print()
    #print('========== Eval Epoch:{} Start =========='.format(epoch))
    epoch_loss = 0  # 单次损失
    mse_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        for data, survival_day, age, resection in loader:
            data, survival_day, age, resection = data.to(device), survival_day.to(
                device), age.to(device), resection.to(device)
            output = model(data, age, resection)
            #print(output.item(), survival_day.item())
            pred, truth = output.item(), survival_day.item()
            preds.append(pred)
            truths.append(truth)
            loss = criterion(output, survival_day)
            mse = val_criterion(output, survival_day).item()
            mse_loss += mse
            loss_ = loss.item()
            epoch_loss += loss_
            #print(loss_, mse)
        preds = dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
    #print(truths, preds)
    print(f"MAE = {mean_absolute_error(truths, preds)}, "
          f"RMSE = {mean_squared_error(truths, preds)**0.5}")
    mae = mean_absolute_error(truths, preds)
    rsme = mean_squared_error(truths, preds)**0.5
    all_ = [(truth, pred) for truth, pred in zip(truths, preds)]
    from operator import itemgetter
    all_.sort(key=itemgetter(0))

    c_index = concordance_index(truths, preds)

    preds, truths = preds.flatten(), truths.flatten()
    pearsonr_value, p_value = pearsonr(preds, truths)
    #print(f"PearsonR: {statistic=}, {p_value=}")
    spearmanr_value, p_value = spearmanr(preds, truths)
    #print(f"SpearmanR: {statistic=}, {p_value=}")
    """plt.scatter(truths, preds, alpha=0.5, label='Data Points')
    plt.plot([0, 1750], [0, 1750], 'r--', label='y=x line')  # 绘制 y=x 参考线
    # 添加图形标签和标题
    plt.title('Scatter Plot of Actual vs Predicted Survival Times')
    plt.xlabel('Truth Survival Time')
    plt.ylabel('Predicted Survival Time')
    plt.legend()
    plt.grid(True)"""

    ax = pg.plot_blandaltman(truths, preds, ax=ax)
    ax.set_xlabel("Mean of Survival Days")
    ax.set_ylabel("Difference of Survival Days")
    ax.set_xlim(0, 1400)
    ax.set_xticks(range(0, 1500, 100))
    ax.set_xticklabels(
        map(str, (0, "", 200, "", 400, "", 600, "", 800, "", 1000, "", 1200, "", 1400)))
    ax.set_ylim(-1000, 1500)
    ax.set_yticks(range(-1000, 1600, 250))
    ax.set_yticklabels(map(str, (-1000, "", -500, "", 0, "", 500, "", 1000, "", 1500)))
    ax.xaxis.label.set_fontsize(24)
    ax.yaxis.label.set_fontsize(24)
    """kmf = KaplanMeierFitter()
    kmf.fit(durations=truths, event_observed=[1 for i in truths], label="Truth")
    kmf.plot_survival_function(ax=ax)
    kmf.fit(durations=preds, event_observed=[1 for i in preds], label="Predict")
    kmf.plot_survival_function(ax=ax)"""

    all_data = pd.DataFrame({
        "time": truths.tolist() + preds.tolist(),
        "event": [1] * len(truths) + [1] * len(preds),
        "type": [0] * len(truths) + [1] * len(preds)
    })
    cph = CoxPHFitter()
    cph.fit(all_data, duration_col='time', event_col='event')
    variable = 'type'
    #p_value = cph.summary.loc["pred", 'p']
    p_value = logrank_test(truths, preds).p_value
    hr = cph.summary.loc["type", 'exp(coef)']
    ci_lower = cph.summary.loc[variable, 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc[variable, 'exp(coef) upper 95%']
    return c_index, mae, rsme, spearmanr_value, ax, p_value, hr, f"[{ci_lower:.4f},{ci_upper:.4f}]"


if __name__ == "__main__":
    import csv
    from pathlib import Path
    learning_rate = ["2.5e-5", "5e-5", "", "2e-4", "4e-4"]
    content = []
    with open("0_diff_lr.csv", "w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Optim", "TraditionModel", "MAE", "RMSE", "C-index", "SpearmanR", "p-value",
            "hr", "95%-ci"
        ])
        for l in learning_rate:
            if l:
                l = f"_{l}"
            p = Path(f"Only_3d_4{l}")
            #fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)
            model = torch.load(p / "Rprop/model/best_epoch.pth").cuda()
            c_index, mae, rsme, spearmanr_value, ax, p_value, hr, ci = val()
            writer.writerow([
                l, (p / "Rprop/best_epoch.txt").read_text(), mae, rsme, c_index,
                spearmanr_value, p_value, hr, ci
            ])
            #plt.savefig(rf"Only_3d_4/{op.name}.pdf", bbox_inches='tight')
            #fig.suptitle(f'Bland–Altman plot with Different Optimizer')
            #plt.savefig(f"{op}_kmf.png")
            #plt.cla()

    #plt.savefig("2_testing.jpg", bbox_inches='tight')

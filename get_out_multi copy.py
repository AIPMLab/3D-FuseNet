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

device = "cuda:0"

#epoch = int(sys.argv[1]) if sys.argv[1:] else 99
#model = torch.load(rf"model/epoch_{epoch}.pth").cuda()
#print(type(model))

criterion = nn.L1Loss().cuda()
val_criterion = nn.MSELoss().cuda()


def val():
    # 1. 训练数据集
    loader = dataloader
    # 2. 进入训练模式
    model.eval()
    #print()
    #print('========== Eval Epoch:{} Start =========='.format(epoch))
    epoch_loss = 0  # 单次损失
    mse_loss = 0
    dice_loss = 0
    preds, truths = [], []
    from monai import losses
    criterion_segment = losses.DiceLoss(to_onehot_y=False, sigmoid=True).cuda()
    with torch.no_grad():
        for data, segment, survival_day, age, resection in loader:
            data, survival_day, age, resection = data.to(device), survival_day.to(
                device), age.to(device), resection.to(device)
            segment = segment.to(device)
            seg, output = model(data)
            segment_loss = criterion_segment(seg, segment)
            #print(output.item(), survival_day.item())
            pred, truth = output.item(), survival_day.item()
            preds.append(pred)
            truths.append(truth)
            loss = criterion(output, survival_day)
            mse = val_criterion(output, survival_day).item()
            mse_loss += mse
            loss_ = loss.item()
            epoch_loss += loss_
            s = segment_loss.item()
            dice_loss += 1 - s
            print(s)
            #print(loss_, mse)
        preds = dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
    #print(truths, preds)
    print(f"Dice = {dice_loss / len(loader):.2%},")
    print(f"MAE = {mean_absolute_error(truths, preds)}, "
          f"RMSE = {mean_squared_error(truths, preds)**0.5}")
    mae = mean_absolute_error(truths, preds)
    rsme = mean_squared_error(truths, preds)**0.5
    all_ = [(truth, pred) for truth, pred in zip(truths, preds)]
    from operator import itemgetter
    all_.sort(key=itemgetter(0))

    c_index = concordance_index(truths, preds)

    preds, truths = preds.flatten(), truths.flatten()
    print(preds)
    print(truths)
    pearsonr_value, p_value = pearsonr(preds, truths)
    #print(f"PearsonR: {statistic=}, {p_value=}")
    spearmanr_value, p_value = spearmanr(preds, truths)
    #print(f"SpearmanR: {statistic=}, {p_value=}")
    """all_data = pd.DataFrame({
        "time": truths,
        "event": [1] * len(truths),
        "pred": preds,
    })
    cph = CoxPHFitter()
    cph.fit(all_data, duration_col='time', event_col='event')
    variable = 'pred'
    p_value = cph.summary.loc["pred", 'p']
    hr = cph.summary.loc["pred", 'exp(coef)']
    ci_lower = cph.confidence_intervals_.loc[variable, '95% lower-bound']
    ci_upper = cph.confidence_intervals_.loc[variable, '95% upper-bound']
    plt.scatter(truths, preds, alpha=0.5, label='Data Points')
    plt.plot([0, 1750], [0, 1750], 'r--', label='y=x line')  # 绘制 y=x 参考线
    # 添加图形标签和标题
    plt.title('Scatter Plot of Actual vs Predicted Survival Times')
    plt.xlabel('Truth Survival Time')
    plt.ylabel('Predicted Survival Time')
    plt.legend()
    plt.grid(True)"""

    ax = pg.plot_blandaltman(truths, preds)
    """kmf = KaplanMeierFitter()
    kmf.fit(durations=truths,
            event_observed=[1 for i in truths],
            label="Truth")
    ax = kmf.plot_survival_function()
    kmf.fit(durations=preds,
            event_observed=[1 for i in preds],
            label="Predict")
    kmf.plot_survival_function(ax=ax)"""
    pd.DataFrame({
        "y_true": truths,
        "y_pred": preds
    }).to_csv(rf"combination_exp_multi/{op.stem}.csv")
    return c_index, mae, rsme, pearsonr_value, spearmanr_value, ax,  #p_value, hr, ci_lower, ci_upper


if __name__ == "__main__":
    import csv
    from pathlib import Path

    #for i in range(5):

    dataset = BraTSDataset(
        r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
        "test",
        using_seg=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    p = Path(f"combination_exp_multi_freeze")
    p.mkdir(exist_ok=True)
    with open(p / f"0_0_testing.csv", "w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Optimizer", "Best Epoch", "MAE", "RSME", "C index", "PearsonR", "SpearmanR"
        ])
        #for op in track(list(p.iterdir())):
        #if not op.is_dir() or op.name.startswith("_"): continue
        op = p
        print(op.name, (op / "best_epoch.txt").read_text())
        model = torch.load(op / "model/best_epoch.pth").cuda()
        c_index, mae, rsme, pearsonr_value, spearmanr_value, ax = val()
        writer.writerow([
            op.name,
            (op / "best_epoch.txt").read_text(),
            mae,
            rsme,
            c_index,
            spearmanr_value,  #p_value, hr,
            #f"[{ci_lower:.4f},{ci_upper:.4f}]"
        ])
        #plt.title(f'KM plot with {op.name}')
        #plt.savefig(f"{op}_ba.png")
        #plt.cla()

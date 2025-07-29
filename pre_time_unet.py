import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from dataloader import BraTSDataset
from image_encoder3D import VitRegressionModel

device = "cuda:0"

dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "test",
    using_seg=False,
    using_mri=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

epoch = int(sys.argv[1]) if sys.argv[1:] else 99
model = torch.load(rf"D:\vit_reg\optimizer_unet\SGD\model\best_epoch.pth").cuda()
print(type(model))

criterion = nn.L1Loss().cuda()
val_criterion = nn.MSELoss().cuda()


def test(epoch: int):
    # 1. 训练数据集
    loader = dataloader
    # 2. 进入训练模式
    model.eval()
    print()
    print('========== Eval Epoch:{} Start =========='.format(epoch))
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
        #print(f"MAE = {epoch_loss / len(loader)}, RMSE = {(mse_loss / len(loader))**0.5}")
        preds = dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
    print(truths, preds)
    print(f"MAE = {mean_absolute_error(truths, preds)}, "
          f"RMSE = {mean_squared_error(truths, preds)**0.5}")
    all_ = [(truth, pred) for truth, pred in zip(truths, preds)]
    from operator import itemgetter
    all_.sort(key=itemgetter(0))

    print(concordance_index(truths, preds))
    truths = truths[0]
    preds = preds[0]
    #print(preds)
    kmf = KaplanMeierFitter()
    kmf.fit(durations=truths, event_observed=[1 for i in truths], label="Truth")
    ax = kmf.plot_survival_function()
    kmf.fit(durations=preds, event_observed=[1 for i in preds], label="Predict")
    kmf.plot_survival_function(ax=ax)

    plt.savefig("test_2.png")
    from scipy.stats import pearsonr, spearmanr

    statistic, p_value = pearsonr(preds, truths)
    print(f"PearsonR: {statistic=}, {p_value=}")
    statistic, p_value = spearmanr(preds, truths)
    print(f"SpearmanR: {statistic=}, {p_value=}")
    exit()
    plt.cla()
    plt.plot(range(len(preds)), [i for _, i in all_], label="Predict")
    plt.plot(range(len(truths)), [i for i, _ in all_], label="Truths")
    plt.legend()
    plt.savefig("truth.png")

    return epoch_loss / len(loader)


if __name__ == "__main__":
    test(epoch)

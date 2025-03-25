import torch
from torch.utils.data import DataLoader
from dataloader import BraTSDataset
from image_encoder3D import VitRegressionModel
from lifelines.utils import concordance_index
from torch import nn
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
import sys
from scipy.stats import pearsonr, spearmanr
from rich.progress import track
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import numpy as np

device = "cuda:0"

dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "val",
    using_seg=False,
    using_mri=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#epoch = int(sys.argv[1]) if sys.argv[1:] else 99
#model = torch.load(rf"model/epoch_{epoch}.pth").cuda()
#print(type(model))

criterion = nn.L1Loss().cuda()
val_criterion = nn.MSELoss().cuda()


def val(epoch: int):
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
        preds = dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
    #print(truths, preds)
    print(
        f"MAE = {mean_absolute_error(truths, preds)}, "
        f"RMSE = {mean_squared_error(truths, preds)**0.5}"
    )
    mae = mean_absolute_error(truths, preds)
    rsme = mean_squared_error(truths, preds)**0.5
    all_ = [(truth, pred) for truth, pred in zip(truths, preds)]
    from operator import itemgetter
    all_.sort(key=itemgetter(0))

    c_index = concordance_index(truths, preds)


    #statistic, p_value = pearsonr(preds, truths)
    #print(f"PearsonR: {statistic=}, {p_value=}")
    #statistic, p_value = spearmanr(preds, truths)
    #print(f"SpearmanR: {statistic=}, {p_value=}")
    return c_index, mae, rsme

if __name__ == "__main__":
    c_indexs, maes, rsmes = [], [], []
    for epoch in track(range(200)):
        model = torch.load(rf"model/epoch_{epoch}.pth").cuda()
        c_index, mae, rsme = val(epoch)
        c_indexs.append(c_index)
        maes.append(mae)
        rsmes.append(rsme)

    import json
    with open("test.json", "w") as f:
        json.dump({"c-index": c_indexs, "mae": maes, "rsme": rsmes}, f)
import torch
from torch.utils.data import DataLoader
from dataloader_combine import BraTSDataset
from image_encoder3D import VitRegressionModel
from lifelines.utils import concordance_index
from torch import nn
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import numpy as np
from monai import losses
from hausdorff import hausdorff_distance
from medpy.metric.binary import hd95

device = "cuda:0"

dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "test",
    using_seg=False,
    using_mri=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

epoch = int(sys.argv[1]) if sys.argv[1:] else 99
model = torch.load(r"D:\vit_reg\seg_exp2\model\best_epoch.pth").cuda()
print(type(model))

criterion = nn.L1Loss().cuda()
val_criterion = nn.MSELoss().cuda()
criterion_seg = losses.DiceLoss(to_onehot_y=False, sigmoid=True).cuda()

from scipy.spatial.distance import directed_hausdorff

"""def hd95(vol1, vol2):
    # 将 3D 体积转换为点集
    vol1_points = np.argwhere(vol1)
    vol2_points = np.argwhere(vol2)
    
    # 计算从 vol1 到 vol2 的所有 Hausdorff 距离
    distances_1_to_2 = [directed_hausdorff(vol1_points, vol2_points)[0] for _ in range(len(vol1_points))]
    
    # 计算从 vol2 到 vol1 的所有 Hausdorff 距离
    distances_2_to_1 = [directed_hausdorff(vol2_points, vol1_points)[0] for _ in range(len(vol2_points))]
    
    # 合并两组距离
    all_distances = np.concatenate((distances_1_to_2, distances_2_to_1))
    
    # 返回 95% 分位数的距离，即 HD95
    return np.percentile(all_distances, 95)"""

def test(epoch: int):
    # 1. 训练数据集
    loader = dataloader
    # 2. 进入训练模式
    model.eval()
    print()
    print('========== Eval Epoch:{} Start =========='.format(epoch))
    epoch_loss = 0  # 单次损失
    mse_loss = 0
    seg_dice = 0
    preds, truths = [], []
    ta = [0,0,0]
    hd = [0,0,0]
    with torch.no_grad():
        for (data, label, clinical, rad, survival_days) in loader:
            data, label, clinical, rad, survival_days = data.to(
                device), label.to(device), clinical.to(device), rad.to(
                    device), survival_days.to(device)
            segment = model(data)
            diceloss = criterion_seg(segment, label)
            # 应用 sigmoid 激活函数，将 logits 转化为概率
            probs = torch.sigmoid(segment)

            # 定义一个函数来计算单个类的 Dice 系数
            def dice_coefficient(prob, truth, epsilon=1e-6):
                intersection = torch.sum(prob * truth)
                union = torch.sum(prob) + torch.sum(truth)
                dice = (2.0 * intersection + epsilon) / (union + epsilon)
                return dice

            for i in range(3):
                prob = probs[:, i:i+1]
                truth = label[:, i:i+1]
                dice = dice_coefficient(prob, truth)
                ta[i] += dice.item()
                hd[i] += hd95(prob.cpu().numpy(), truth.cpu().numpy())
            seg_dice += 1 - diceloss.item()
    #print(truths, preds)
    print(f"Dice = {seg_dice / len(loader):.2%},")
    print(f"{ta[0] / len(loader):.2%}, {ta[1] / len(loader):.2%}, {ta[2] / len(loader):.2%}")
    print(f"{hd[0] / len(loader):.2%}, {hd[1] / len(loader):.2%}, {hd[2] / len(loader):.2%}")


if __name__ == "__main__":
    test(epoch)

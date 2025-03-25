import torch
from torch.utils.data import DataLoader
from dataloader import BraTSDataset
from image_encoder3D import VitRegressionModel
from lifelines.utils import concordance_index
from torch import nn
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from sklearn.metrics import confusion_matrix
import seaborn as snsC

device = "cuda:0"

dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "test",
    using_seg=False,
    using_mri=True)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False)

model = torch.load(r"model/epoch_15.pth").cuda()

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
            data, survival_day, age, resection = data.to(
                device), survival_day.to(device), age.to(
                    device), resection.to(device)
            output = model(data, age, resection)
            #print(output.item(), survival_day.item())
            pred, truth = output.item(), survival_day.item()
            pred = 0 if pred < 0.5 else 1
            preds.append(pred)
            truths.append(truth)
        cm = confusion_matrix(truths, preds, labels=[0, 1])
        f, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax)  #画热力图

        ax.set_title('confusion matrix')  #标题
        ax.set_xlabel('predict')  #x轴
        ax.set_ylabel('true')  #y轴
        f.savefig("heatmap.png")

        print("accuracy_score", accuracy_score(truths, preds),
              "precision_score", precision_score(truths, preds),
              "recall_score", recall_score(truths, preds), 
              "f1_score", f1_score(truths, preds))
    return epoch_loss / len(loader)


if __name__ == "__main__":
    test(1)

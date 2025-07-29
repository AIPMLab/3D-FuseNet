import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import torch
import torch.optim.sgd
from lifelines.utils import concordance_index
from monai import losses
from rich import print as rprint
from rich.console import Group
from rich.live import Live
from rich.progress import Progress, TextColumn, TimeElapsedColumn, track
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader
from utils import EarlyStopping, SlidingWindowInference

from dataloader import BraTSDataset
from segformer3d import CombinationModel

start_time = time.time()
console = rich.get_console()
Path("model").mkdir(exist_ok=True)
Path("log").mkdir(exist_ok=True)
with console.status("Loading Dataset..."):
    epochs = 200
    torch.manual_seed(42)
    device = "cuda:0"
    train_dataset = BraTSDataset(
        r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
        "train",
        using_seg=True,
        using_mri=True)
    test_dataset = BraTSDataset(
        r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
        "val",
        using_seg=True,
        using_mri=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
rprint(
    f"Loading Dataset... [green]Complete[default], Using {time.time()-start_time:.2f} sec"
)
#model = RegressionModel(with_clinical=False).cuda()
#model.load_state_dict(torch.load("pytorch_model.bin"), strict=False)
#model = torch.load("model/epoch_199.pth")
#for name, param in model.named_parameters():
#    if "MixVisionTransformer" in name:
#        param.requires_grad = False
#model.load_state_dict(torch.load('model/epoch_299.pth').state_dict())

criterion_segment = losses.DiceLoss(to_onehot_y=False, sigmoid=True).cuda()
criterion_survival = nn.HuberLoss().cuda()
val_criterion = nn.MSELoss().cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]).to(device))

#optimizer = torch.optim.SGD(model.parameters(), lr=8e-05)
optimizer_list = [
    #torch.optim.SGD,
    torch.optim.Adadelta,
    torch.optim.Adagrad,
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SparseAdam,
    #torch.optim.Adamax,
    torch.optim.ASGD,
    #torch.optim.LBFGS,
    torch.optim.NAdam,
    torch.optim.RAdam,
    torch.optim.RMSprop,
    torch.optim.Rprop,
]
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,eps=1e-08,)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
#                                                       T_max=5,
#                                                       eta_min=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)
#optimizer.load_state_dict(torch.load('model/optimizer_299.pth').state_dict())

#for name, value in model.named_parameters():
#     if "vit" in name:
#          value.requires_grad = False

train_progress = Progress(*Progress.get_default_columns(),
                          TextColumn("Loss={task.fields[loss]}"))
test_progress = Progress(*Progress.get_default_columns(),
                         TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}"))
test_progress = Progress(*Progress.get_default_columns(),
                         TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}"))
overall_progress = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    TextColumn(
        "Best Epoch={task.fields[best_epoch]}, No Update={task.fields[no_improve]}"),
    speed_estimate_period=300)
progress_group = Group(
    train_progress,
    test_progress,
    overall_progress,
)

train_task = train_progress.add_task("Training", loss=0)


def train(epoch: int):
    # 1. 训练数据集
    loader = train_dataloader
    # 2. 进入训练模式
    model.train()
    print()
    rprint('========== Train Epoch:{} Start =========='.format(epoch))
    epoch_loss = 0  # 单次损失
    p, task = train_progress, train_task
    for i, (data, label, survival_days, age,
            resection) in enumerate(p.track(loader, task_id=task), 1):
        data, label, survival_days, age, resection = data.to(device), label.to(
            device), survival_days.to(device), age.to(device), resection.to(device)
        optimizer1.zero_grad()
        #optimizer2.zero_grad()

        batch_cache = [c.to(device) for c in cache[i]]
        segment, survival = model(data, batch_cache)
        survival_loss = criterion_survival(survival, survival_days)
        #segment_loss = criterion_segment(segment, label)
        loss = survival_loss  #*0.75 + segment_loss *0.25
        loss.backward()
        optimizer1.step()
        #optimizer2.step()
        loss_ = loss.item()
        epoch_loss += loss_
        p.update(task, loss=loss_)
    rprint(f"loss = {epoch_loss / len(loader)}")
    p.update(task, loss=0)
    #with Path(f"log/train_{epoch}.csv").open("w") as f:
    #    f.write("pred, truth\n")
    #    for pred, truth in zip(preds, truths):
    #        f.write(f"{pred}, {truth}\n")
    return epoch_loss / len(loader)


test_task = test_progress.add_task("Testing", loss=0, mse=0)


def test(epoch: int):
    # 1. 训练数据集
    loader = test_dataloader
    # 2. 进入训练模式
    model.eval()
    print()
    print('========== Eval Epoch:{} Start =========='.format(epoch))
    epoch_loss = 0  # 单次损失
    mse_loss = 0
    dice_loss = 0
    preds, truths = [], []
    with torch.no_grad():
        p, task = test_progress, test_task
        for i, (data, label, survival_days, age,
                resection) in enumerate(p.track(loader, task_id=task), 1):
            data, label, survival_days, age, resection = data.to(device), label.to(
                device), survival_days.to(device), age.to(device), resection.to(device)
            #batch_cache = [c.to(device) for c in cache[i]]
            segment, survival = model(data)
            preds += [pred.item() for pred in survival]
            truths += [truth.item() for truth in survival_days]
            survival_loss = criterion_survival(survival, survival_days)
            segment_loss = criterion_segment(segment, label)
            loss = survival_loss * 0.75  # + segment_loss * 0.25
            mse = val_criterion(survival, survival_days).item()
            mse_loss += mse
            loss_ = loss.item()
            epoch_loss += loss_
            dice_loss += 1 - segment_loss.item()
            p.update(task, loss=epoch_loss / i, mse=mse_loss / i)
        preds = test_dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = test_dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
        rprint(f"loss = {epoch_loss / len(loader)}")
        rprint(f"MAE = {mean_absolute_error(truths, preds):4f}, "
               f"RMSE = {mean_squared_error(truths, preds)**0.5:4f}, "
               f"C index = {concordance_index(truths, preds):8f},"
               f"Dice = {dice_loss / len(loader):.2%},")
        #preds = [int(pred > 0.5) for pred in preds]
        #print(preds)
        #print(truths)
        #acc = accuracy_score(truths, preds)
        #recall = recall_score(truths, preds)
        #f1 = f1_score(truths, preds)
        #rprint(f"{acc=}, {recall=}, {f1=}")
        p.update(task, loss=0, mse=0)
        #with Path(f"log/val_{epoch}.csv").open("w") as f:
        #    f.write("pred, truth\n")
        #    for pred, truth in zip(preds[0], truths[0]):
        #        f.write(f"{pred}, {truth}\n")
    return epoch_loss / len(loader), concordance_index(truths, preds)


no_improve = 0
best_epoch = ...
if __name__ == '__main__':
    experimance_path = Path(__file__).parent / "combination_exp_multi"

    #model.load_state_dict(pretrained_model.state_dict())
    #model.survival_coder.output_mlp.load_state_dict(pretrained_model.survival_coder.output_mlp.state_dict())
    #model.survival_coder.clinical_mlp.load_state_dict(pretrained_model.survival_coder.clinical_mlp.state_dict())
    #model.survival_coder.mlp.load_state_dict(pretrained_model.survival_coder.mlp.state_dict(), strict=False)

    #optimizer2 = torch.optim.SGD([
    #    {'params': model.segformer_encoder.parameters()},
    #    {'params': model.survival_coder.parameters()},
    #], lr=0.001)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)

    with Live(progress_group):
        for op_num, op in enumerate(optimizer_list, 1):
            all_train_acc = []
            all_test_acc = []
            all_test_cindex = []
            model = CombinationModel().cuda()
            pretrained_model = torch.load(r"D:\vit_reg\seg_exp2\model\best_epoch.pth")
            model.segformer_encoder.load_state_dict(
                pretrained_model.segformer_encoder.state_dict())
            model.segformer_decoder.load_state_dict(
                pretrained_model.segformer_decoder.state_dict())
            model.freeze_encoder()

            # 首轮生成缓存
            cache = {}
            model.train()
            for i, (data, label, survival_days, age,
                    resection) in enumerate(train_dataloader, 1):
                data = data.to(device)
                with torch.no_grad():
                    encoded = model.segformer_encoder(data)
                    decoded = model.segformer_decoder(*encoded)
                cache[i] = [
                    *[encoded[j].detach().cpu() for j in range(4)],
                    decoded.detach().cpu()
                ]  # 保存到CPU内存
            print(cache.keys())
            task = overall_progress.add_task(
                f"[{op_num}/{len(optimizer_list)}]Total({op.__name__})",
                total=epochs,
                best_epoch=best_epoch,
                no_improve=no_improve)
            op_folder = experimance_path / op.__name__
            (op_folder / f"model").mkdir(parents=True, exist_ok=True)
            optimizer1 = op([
                #{'params': model.segformer_encoder.parameters(), "lr": 2e-06},
                #{'params': model.segformer_decoder.parameters(), "lr": 1e-04},
                {
                    'params': model.decoder_model.parameters(),
                    "lr": 1e-04
                },
                #{'params': model.survival_coder.parameters(), "lr": 1e-04},
            ])  #调整自己想要的学习率
            for epoch in range(epochs):
                trainAcc = train(epoch)
                valAcc, c_index = test(epoch)
                all_train_acc.append(trainAcc)
                all_test_acc.append(valAcc)
                all_test_cindex.append(c_index)
                #scheduler.step()

                if len(all_test_acc) <= 1 or valAcc <= min(all_test_acc[:-1]):
                    torch.save(model, op_folder / f"model/best_epoch.pth")
                    torch.save(optimizer1, op_folder / f"model/best_optimizer1.pth")
                    #torch.save(
                    #    optimizer2, experimance_path /
                    #    f"model/best_optimizer2.pth")
                    rprint(f"Best Model Update, Epoch {epoch}")
                    (op_folder / f"best_epoch.txt").write_text(str(epoch))
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    rprint(f"No improve update: {no_improve}")

                overall_progress.update(task,
                                        best_epoch=best_epoch,
                                        no_improve=no_improve)

                if no_improve >= 50:
                    rprint("[red]EarlyStopping[default]")
                    break

                with (op_folder / f"test.csv").open("w") as f:
                    for i, j, k in zip(all_train_acc, all_test_acc, all_test_cindex):
                        f.write(f"{i},{j},{k}\n")
                train_progress.reset(train_task)
                test_progress.reset(test_task)
                overall_progress.advance(task)
            pd.read_csv(op_folder / f"test.csv",
                        names=["Train Loss", "Val Loss",
                               "Val C-index"]).plot(figsize=(15, 5),
                                                    xticks=range(0, 205, 5),
                                                    title=f"Loss",
                                                    xlabel="Epoch",
                                                    ylabel="Loss")
            plt.savefig(op_folder / f"loss.png")
            plt.cla()
            plt.close()

        overall_progress.remove_task(task)
"""
if __name__ == '__main__':
    all_op = len(optimizer_list)
    experimance_path = Path("optimizer_test_clinical_multi_avgpooling")
    for op_num, op in enumerate(optimizer_list, 1):
        all_train_acc = []
        all_test_acc = []
        all_test_cindex = []
        op_folder = experimance_path / op.__name__
        model = RegressionModel(with_clinical=True).cuda()
        #model_path = op_folder / "model/best_epoch.pth"
        #if model_path.exists():
        #    m = torch.load(model_path).state_dict()
        #    model.load_state_dict(m)
        #model.load_state_dict(torch.load("pytorch_model.bin"), strict=False)
        loss = nn.HuberLoss()
        optimizer = op(model.parameters(), lr=1e-04)
        task = overall_progress.add_task(f"[{op_num}/{all_op}]Total({op.__name__})", total=epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)
        with Live(progress_group):
            (op_folder / "model").mkdir(parents=True, exist_ok=True)
            for epoch in range(epochs):
                trainAcc = train(epoch)
                valAcc, c_index = test(epoch)
                all_train_acc.append(trainAcc)
                all_test_acc.append(valAcc)
                all_test_cindex.append(c_index)
                rprint(
                    f"Learning Rate in Epoch {epoch}: {optimizer.param_groups[0]['lr']}"
                )
                scheduler.step()
                if len(all_test_acc) <= 1 or valAcc <= min(all_test_acc[:-1]):
                    torch.save(model, op_folder / "model/best_epoch.pth")
                    torch.save(optimizer, op_folder / "model/best_optimizer.pth")
                    rprint(f"Best Model Update, Epoch {epoch}")
                    (op_folder / "best_epoch.txt").write_text(str(epoch))
                with (op_folder / "test.csv").open("w") as f:
                    for i, j, k in zip(all_train_acc, all_test_acc, all_test_cindex):
                        f.write(f"{i},{j},{k}\n")
                train_progress.reset(train_task)
                test_progress.reset(test_task)
                overall_progress.advance(task)
            pd.read_csv(op_folder / "test.csv",
                names=["Train Loss", "Val Loss", "Val C-index"
                    ]).plot(figsize=(15, 5),
                            xticks=range(0, 205, 5),
                            title=f"Loss with {op.__name__}",
                            xlabel="Epoch",
                            ylabel="Loss")
            plt.savefig(op_folder / "loss.png")
            plt.cla()
            plt.close()

            overall_progress.remove_task(task)
"""

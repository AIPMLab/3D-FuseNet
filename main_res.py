import torch
from torch.utils.data import DataLoader
from dataloader import BraTSDataset
from resnet import RegressionModel
from torch import nn
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from pathlib import Path
from rich import print as rprint
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, recall_score, f1_score
from rich.console import Group
from rich.live import Live
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Path("model").mkdir(exist_ok=True)
Path("log").mkdir(exist_ok=True)

torch.backends.cudnn.benchmark = True

epochs = 200
torch.manual_seed(42)
device = "cuda:0"
train_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "train",
    using_seg=False,
    using_mri=True)
test_dataset = BraTSDataset(
    r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    "val",
    using_seg=False,
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
#model = RegressionModel(with_clinical=False).cuda()
#model.load_state_dict(torch.load("pytorch_model.bin"), strict=False)
#model = torch.load("model/epoch_199.pth")
#for name, param in model.named_parameters():
#    if "MixVisionTransformer" in name:
#        param.requires_grad = False
#model.load_state_dict(torch.load('model/epoch_299.pth').state_dict())

criterion = nn.HuberLoss().cuda()
val_criterion = nn.MSELoss().cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]).to(device))

#optimizer = torch.optim.SGD(model.parameters(), lr=8e-05)
optimizer_list = [
    
#torch.optim.SGD,
#torch.optim.Adadelta,
#torch.optim.Adagrad,
torch.optim.Adam,
#torch.optim.AdamW,
#torch.optim.SparseAdam,
#torch.optim.Adamax,
#torch.optim.ASGD,
#torch.optim.LBFGS,
#torch.optim.NAdam,
#torch.optim.RAdam,
#torch.optim.RMSprop,
#torch.optim.Rprop,
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
test_progress = Progress(
    *Progress.get_default_columns(),
    TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}"))
test_progress = Progress(
    *Progress.get_default_columns(),
    TextColumn("Loss={task.fields[loss]}, MSE={task.fields[mse]}"))
overall_progress = Progress(*Progress.get_default_columns(),
                            TimeElapsedColumn(),
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
    preds, truths = [], []
    p, task = train_progress, train_task
    for i, (data, survival_day, age,
            resection) in enumerate(p.track(loader, task_id=task)):
        data, survival_day, age, resection = data.to(
            device), survival_day.to(device), age.to(
                device), resection.to(device)
        optimizer.zero_grad()
        output = model(data)
        preds += [pred.item() for pred in output]
        truths += [truth.item() for truth in survival_day]
        loss = criterion(output, survival_day)
        loss.backward()
        optimizer.step()
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
    preds, truths = [], []
    with torch.no_grad():
        p, task = test_progress, test_task
        for i, (data, survival_day, age,
                resection) in enumerate(p.track(loader, task_id=task),
                                        1):
            data, survival_day, age, resection = data.to(
                device), survival_day.to(device), age.to(
                    device), resection.to(device)
            output = model(data)
            preds += [
                pred.item()
                for pred in output
            ]
            truths += [
                truth.item()
                for truth in survival_day
            ]
            loss = criterion(output, survival_day)
            mse = val_criterion(output, survival_day).item()
            mse_loss += mse
            loss_ = loss.item()
            epoch_loss += loss_
            p.update(task, loss=epoch_loss / i, mse=mse_loss / i)
        preds = test_dataset.scaler.inverse_transform(np.array(preds).reshape(1, -1))
        truths = test_dataset.scaler.inverse_transform(np.array(truths).reshape(1, -1))
        rprint(
            f"MAE = {mean_absolute_error(truths, preds)}, "
            f"RMSE = {mean_squared_error(truths, preds)**0.5}, "
            f"C index = {concordance_index(truths, preds)}"
        )
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

from utils import EarlyStopping
"""if __name__ == '__main__':
    all_train_acc = []
    all_test_acc = []
    all_test_cindex = []
    experimance_path = Path("optimizer_test_clinical")
    model = RegressionModel(with_clinical=True).cuda()
    #seg = torch.load("pytorch_model.bin")
    #seg = {k.replace('segformer_encoder.', ''): v for k, v in seg.items() if k.startswith("segformer_encoder.")}
    #model.encoder_model.load_state_dict(seg)
    #for name, value in model.named_parameters():
    # if "encoder_model." in name:
    #      value.requires_grad = False
    loss = nn.HuberLoss()
    op = torch.optim.ASGD
    optimizer = op(model.parameters(), lr=8e-05)
    task = overall_progress.add_task(f"Total", total=epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.8)
    with Live(progress_group):
        (experimance_path / f"{op.__name__}/model").mkdir(parents=True, exist_ok=True)
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
                torch.save(model, experimance_path / f"{op.__name__}/model/bset_epoch.pth")
                torch.save(optimizer, experimance_path / f"{op.__name__}/model/best_optimizer.pth")
                rprint(f"Best Model Update, Epoch {epoch}")
                (experimance_path / f"{op.__name__}/best_epoch.txt").write_text(str(epoch))
            with (experimance_path / f"{op.__name__}/test.csv").open("w") as f:
                for i, j, k in zip(all_train_acc, all_test_acc, all_test_cindex):
                    f.write(f"{i},{j},{k}\n")
            train_progress.reset(train_task)
            test_progress.reset(test_task)
            overall_progress.advance(task)
        pd.read_csv(experimance_path / f"{op.__name__}/test.csv",
            names=["Train Loss", "Val Loss", "Val C-index"
                ]).plot(figsize=(15, 5),
                        xticks=range(0, 205, 5),
                        title=f"Loss with {op.__name__}",
                        xlabel="Epoch",
                        ylabel="Loss")
        plt.savefig(experimance_path / f"{op.__name__}/loss.png")
        plt.cla()
        plt.close()

        overall_progress.remove_task(task)"""


if __name__ == '__main__':
    torch.cuda.empty_cache()
    all_op = len(optimizer_list)
    experimance_path = Path("resnet50_test")
    for op_num, op in enumerate(optimizer_list, 1):
        all_train_acc = []
        all_test_acc = []
        all_test_cindex = []
        op_folder = experimance_path / op.__name__
        model = RegressionModel(with_clinical=True).cuda()
        model_path = op_folder / "model/best_epoch.pth"
        if model_path.exists():
            m = torch.load(model_path).state_dict()
            model.load_state_dict(m)
        #model.load_state_dict(torch.load("pytorch_model.bin"), strict=False)
        early_stopping = EarlyStopping(patience=10, trace_func=rprint)
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
                #scheduler.step()
                #early_stopping(valAcc)
                #if early_stopping.early_stop:
                #    break
                if len(all_test_acc) <= 1 or valAcc <= min(all_test_acc[:-1]):
                    torch.save(model, op_folder / "model/best_epoch.pth")
                    torch.save(optimizer, op_folder / "model/best_optimizer.pth")
                    rprint(f"[blue]Best Model Update[reset], Epoch {epoch}")
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

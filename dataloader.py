import random
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from torch.nn import functional as F
from torch.utils.data import Dataset

random.seed(42)


def stratified_split_sorted(df, y_col, train_size, val_size, test_size, n_splits=10):
    # 按y值排序
    df = df.sort_values(by=y_col).reset_index(drop=True)

    # 创建一个新列，将y值分为n_splits个分层
    df['y_bin'] = pd.qcut(df.index, q=n_splits, labels=False)

    # 分割数据集
    train, temp = train_test_split(df,
                                   test_size=(1 - train_size),
                                   stratify=df['y_bin'],
                                   random_state=42)
    val, test = train_test_split(temp,
                                 test_size=(test_size / (val_size + test_size)),
                                 stratify=temp['y_bin'],
                                 random_state=42)

    # 删除辅助列
    train = train.drop(columns=['y_bin'])
    val = val.drop(columns=['y_bin'])
    test = test.drop(columns=['y_bin'])

    return train, val, test


class BraTSDataset(Dataset):

    def __init__(
        self,
        folder: str | PathLike,
        data_type: Literal["train", "val", "test"] = "train",
        size: int = 256,
        using_mri: bool = True,
        using_seg: bool = False,
    ):
        self.folder = Path(folder)
        self.data_folder = Path(
            r"D:\vit_reg\SegFormer3D\data\brats2021_seg\BraTS2020_Training_Data_2")
        self.size = size
        """label_data = pandas.read_csv(self.folder /
                                     "survival_info.csv")
        label_data = label_data.sort_values("Survival_days")#[label_data["Survival_days"] > 500]
        resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
        #median_value = 500
        #label_data["Survival_days"] = label_data["Survival_days"].apply(lambda x: 1 if x > median_value else 0)

        label_data["Extent_of_Resection"] = label_data[
            "Extent_of_Resection"].map(resection_list)
        label_data[["ER1", "ER2", "ER3"]] = LabelBinarizer().fit_transform(label_data["Extent_of_Resection"].to_numpy())
        self.using_mri = using_mri
        
        train_data, val_data, test_data = stratified_split_sorted(
            label_data, "Survival_days", 0.7, 0.1, 0.2, n_splits=10)"""
        self.using_seg = using_seg
        train_data = pd.read_csv("data_split/train.csv")

        label_data = pd.read_csv(f"data_split/{data_type}.csv")
        #print(train_data)
        self.scaler = StandardScaler()
        #self.scaler.fit([[0], [1000]])
        self.scaler.fit(train_data["Survival_days"].to_numpy().reshape(-1, 1))

        label_data["Survival_days"] = self.scaler.transform(
            label_data["Survival_days"].to_numpy().reshape(-1, 1))
        #print(label_data["Survival_days"] )
        #age_scaler = StandardScaler()
        age_scaler = MinMaxScaler()
        age_scaler.fit([[0], [100]])
        #age_scaler.fit(train_data["Age"].to_numpy().reshape(-1, 1))
        label_data["Age"] = age_scaler.transform(label_data["Age"].to_numpy().reshape(
            -1, 1))
        self.random_index = label_data["Brats20ID"].tolist()
        self.items = [(self.folder / i) for i in self.random_index]

        self.survival_days = [
            torch.Tensor(
                [label_data[label_data["Brats20ID"] == i]["Survival_days"].values[0]])
            for i in self.random_index
        ]
        self.ages = [
            torch.Tensor([label_data[label_data["Brats20ID"] == i]["Age"].values[0]])
            for i in self.random_index
        ]
        self.resection = [
            torch.Tensor(label_data[label_data["Brats20ID"] == i][["ER1", "ER2",
                                                                   "ER3"]].values[0])
            for i in self.random_index
        ]
        self.images = [
            torch.load(self.data_folder / name / f"{name}_modalities.pt")
            for name in self.random_index
        ]

    def __getitem__(self, index):
        name = self.random_index[index]
        #t1gd_image[seg_image == 0] = 0
        #flair_image[seg_image == 0] = 0

        #image = torch.load(self.data_folder / name /
        #                   f"{name}_modalities.pt")
        ret = [
            self.images[index], self.survival_days[index], self.ages[index],
            self.resection[index]
        ]
        if self.using_seg:
            seg = torch.load(self.data_folder / name / f"{name}_label.pt")
            ret.insert(1, torch.Tensor(seg))
        return ret

    def __len__(self):
        return len(self.items)


class BraTSDataset_Old(Dataset):

    def __init__(
        self,
        folder: str | PathLike,
        data_type: Literal["train", "val", "test"] = "train",
        size: int = 256,
        using_mri: bool = True,
        using_seg: bool = False,
    ):
        self.folder = Path(folder)
        self.size = size
        label_data = pandas.read_csv(self.folder / "survival_info.csv")
        label_data = label_data.sort_values("Survival_days")
        resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
        #median_value = label_data["Survival_days"].median()
        #label_data["Survival_days"] = label_data["Survival_days"].apply(lambda x: 1 if x > median_value else 0)

        label_data["Extent_of_Resection"] = label_data["Extent_of_Resection"].map(
            resection_list)
        label_data[["ER1", "ER2", "ER3"]] = LabelBinarizer().fit_transform(
            label_data["Extent_of_Resection"].to_numpy())
        self.using_mri = using_mri
        self.using_seg = using_seg
        train_data, val_data, test_data = stratified_split_sorted(
            label_data, "Survival_days", 0.7, 0.05, 0.25)
        #print(train_data)
        if data_type == "train":
            self.random_index = train_data["Brats20ID"].tolist()
        elif data_type == "val":
            self.random_index = val_data["Brats20ID"].tolist()
        else:
            self.random_index = test_data["Brats20ID"].tolist()
        self.items = [(self.folder / i) for i in self.random_index]

        self.survival_days = [
            label_data[label_data["Brats20ID"] == i]["Survival_days"].values[0]
            for i in self.random_index
        ]
        self.ages = [
            label_data[label_data["Brats20ID"] == i]["Age"].values[0]
            for i in self.random_index
        ]
        self.resection = [
            label_data[label_data["Brats20ID"] == i]["Extent_of_Resection"].values[0]
            for i in self.random_index
        ]

    def __getitem__(self, index):
        path = self.items[index]
        mri_type = ["*flair*", "*t1ce*", "*t1.nii*", "*t2*"]
        #mri_type = ["*flair*", "*t1ce*", "*t2*"]
        all_image = []
        for i in mri_type:
            sitk_image = sitk.ReadImage(next(path.rglob(i)))
            # Non-zero [(0, 130), (29, 150), (40, 169)]
            image_array = sitk.GetArrayFromImage(sitk_image)[:128, 27:155, 40:168]
            all_image.append(image_array)
        try:
            seg = sitk.ReadImage(next(path.rglob("*seg*")))
        except StopIteration:
            seg = sitk.ReadImage(next(path.rglob("*GlistrBoost*")))

        #t1gd_image[seg_image == 0] = 0
        #flair_image[seg_image == 0] = 0
        seg_image = sitk.GetArrayFromImage(seg).astype(np.uint8)

        image = np.stack(all_image).astype(np.int16)
        ret = [
            torch.Tensor(image),
            torch.Tensor([self.survival_days[index]]),
            torch.Tensor([self.ages[index]]),
            torch.Tensor([self.resection[index]])
        ]
        if self.using_seg:
            ret.insert(1, torch.Tensor(seg_image))
        return ret

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":

    folder = Path(r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    label_data = pandas.read_csv(folder / "survival_info.csv")
    label_data = label_data.sort_values("Survival_days")
    resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
    label_data["Extent_of_Resection"] = label_data["Extent_of_Resection"].map(
        resection_list)
    label_data[["ER1", "ER2", "ER3"]] = LabelBinarizer().fit_transform(
        label_data["Extent_of_Resection"].to_numpy())
    print(label_data[["ER1", "ER2", "ER3"]])
    print(label_data[label_data["Brats20ID"] == "BraTS20_Training_003"][[
        "ER1", "ER2", "ER3"
    ]].values)
    exit()
    resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
    #median_value = label_data["Survival_days"].median()
    #label_data["Survival_days"] = label_data["Survival_days"].apply(lambda x: 1 if x > median_value else 0)

    label_data["Extent_of_Resection"] = label_data["Extent_of_Resection"].map(
        resection_list)
    train_data, val_data, test_data = stratified_split_sorted(
        label_data, "Survival_days", 0.7, 0.05, 0.25)
    print(val_data)

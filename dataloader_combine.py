from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from os import PathLike
import SimpleITK as sitk
import numpy as np
import random
import pandas
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer

random.seed(42)

def stratified_split_sorted(df, y_col, train_size, val_size, test_size, n_splits=10):
    # 按y值排序
    df = df.sort_values(by=y_col).reset_index(drop=True)

    # 创建一个新列，将y值分为n_splits个分层
    df['y_bin'] = pandas.qcut(df.index, q=n_splits, labels=False)

    # 分割数据集
    train, temp = train_test_split(df, test_size=(1 - train_size), stratify=df['y_bin'], random_state=42)
    val, test = train_test_split(temp, test_size=(test_size / (val_size + test_size)), stratify=temp['y_bin'], random_state=42)

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
            r"D:\vit_reg\SegFormer3D\data\brats2021_seg\BraTS2020_Training_Data_2"
        )
        self.size = size
        label_data = pandas.read_csv(self.folder /
                                     "survival_info.csv")
        label_data = label_data.sort_values("Survival_days")#[label_data["Survival_days"] > 500]
        resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
        #median_value = 500
        #label_data["Survival_days"] = label_data["Survival_days"].apply(lambda x: 1 if x > median_value else 0)

        label_data["Extent_of_Resection"] = label_data[
            "Extent_of_Resection"].map(resection_list)
        label_data[["ER1", "ER2", "ER3"]] = LabelBinarizer().fit_transform(label_data["Extent_of_Resection"].to_numpy())
        self.using_mri = using_mri
        self.using_seg = using_seg
        train_data, val_data, test_data = stratified_split_sorted(
            label_data, "Survival_days", 0.7, 0.1, 0.2, n_splits=10)
        #print(train_data)
        self.scaler = StandardScaler()
        #self.scaler.fit([[0], [1000]])
        self.scaler.fit(train_data["Survival_days"].to_numpy().reshape(-1, 1))
        label_data["Survival_days"] = self.scaler.transform(label_data["Survival_days"].to_numpy().reshape(-1, 1))
        #print(label_data["Survival_days"] )
        #age_scaler = StandardScaler()
        age_scaler = MinMaxScaler()
        age_scaler.fit([[0], [100]])
        #age_scaler.fit(train_data["Age"].to_numpy().reshape(-1, 1))
        label_data["Age"] = age_scaler.transform(label_data["Age"].to_numpy().reshape(-1, 1))
        if data_type == "train":
            self.random_index = train_data["Brats20ID"].tolist()
        elif data_type == "val":
            self.random_index = val_data["Brats20ID"].tolist()
        else:
            self.random_index = test_data["Brats20ID"].tolist()
        self.items = [(self.folder / i) for i in self.random_index]

        rad_df = pandas.read_csv(f"radiomics_feature_pre_{data_type}.csv")
        rad_train = pandas.read_csv(f"radiomics_feature_pre_train.csv")
        coloumn_use_rad = rad_df.columns.tolist()[1:-1]
        rad_df = rad_df.fillna(0)
        
        rad_scaler = StandardScaler()
        rad_scaler.fit(rad_train[coloumn_use_rad])
        rad_df[coloumn_use_rad] = rad_scaler.transform(rad_df[coloumn_use_rad])



        self.survival_days = [
            torch.Tensor([
                label_data[label_data["Brats20ID"] == i]
                ["Survival_days"].values[0]
            ]) for i in self.random_index
        ]
        self.clinical = [
            torch.Tensor(
                label_data[label_data["Brats20ID"] == i]
                [["Age", "ER1", "ER2", "ER3"]].values[0]
            ) for i in self.random_index
        ]
        self.images = [
            torch.load(self.data_folder / name /
                       f"{name}_modalities.pt")
            for name in self.random_index
        ]
        self.segment = [
            torch.load(self.data_folder / name /
                       f"{name}_label.pt")
            for name in self.random_index
        ]
        self.rad = [torch.Tensor(
                rad_df[rad_df["name"] == i]
                [coloumn_use_rad].values[0]
            )
            for i in self.random_index]

    def __getitem__(self, index):
        #t1gd_image[seg_image == 0] = 0
        #flair_image[seg_image == 0] = 0

        #image = torch.load(self.data_folder / name /
        #                   f"{name}_modalities.pt")
        ret = [
            self.images[index],self.segment[index], self.clinical[index], self.rad[index], self.survival_days[index],
        ]
        return ret

    def __len__(self):
        return len(self.items)

if __name__ == "__main__":

    folder = Path(r"D:\yolov9\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    label_data = pandas.read_csv(folder / "survival_info.csv")
    label_data = label_data.sort_values("Survival_days")
    resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
    label_data["Extent_of_Resection"] = label_data["Extent_of_Resection"].map(resection_list)
    label_data[["ER1", "ER2", "ER3"]] = LabelBinarizer().fit_transform(label_data["Extent_of_Resection"].to_numpy())
    print(label_data[label_data["Brats20ID"] == "BraTS20_Training_003"][["Age", "ER1", "ER2", "ER3"]].values)

    exit()
    resection_list = {"GTR": 1, "STR": 2, np.nan: 3}
    #median_value = label_data["Survival_days"].median()
    #label_data["Survival_days"] = label_data["Survival_days"].apply(lambda x: 1 if x > median_value else 0)

    label_data["Extent_of_Resection"] = label_data["Extent_of_Resection"].map(resection_list)
    train_data, val_data, test_data = stratified_split_sorted(label_data, "Survival_days", 0.7, 0.05, 0.25)
    print(val_data)

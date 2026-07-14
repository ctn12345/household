# /mnt/sdc/tnchen/matchine_learning/dataset/power_dataset.py

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HouseholdPowerDataset(Dataset):
    """
    用于家庭用电预测任务的数据集。

    支持：
    1. 90 -> 90 短期预测
    2. 90 -> 365 长期预测

    每个样本：
        X: [90, feature_dim]
        y: [pred_len]
    """

    def __init__(
        self,
        root_dir="/mnt/sdc/tnchen/matchine_learning",
        split="train",
        task="90to90",
        return_dates=False
    ):
        """
        Args:
            root_dir: 项目根目录
            split: train 或 eval
            task: 90to90 或 90to365
            return_dates: 是否返回预测日期
        """
        super().__init__()

        assert split in ["train", "eval"], f"split 必须是 train 或 eval，但得到 {split}"
        assert task in ["90to90", "90to365"], f"task 必须是 90to90 或 90to365，但得到 {task}"

        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.return_dates = return_dates

        npz_path = os.path.join(
            root_dir,
            "data",
            split,
            f"{split}_{task}.npz"
        )

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"找不到数据文件: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)

        self.y_dates = data["y_dates"] if "y_dates" in data.files else None
        self.pred_start_dates = data["pred_start_dates"] if "pred_start_dates" in data.files else None

        self.feature_cols = data["feature_cols"].tolist() if "feature_cols" in data.files else None
        self.target_col = data["target_col"].tolist()[0] if "target_col" in data.files else None

        print(f"[Load Dataset] split={split}, task={task}")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        print(f"  feature_dim: {self.X.shape[-1]}")
        print(f"  pred_len: {self.y.shape[-1]}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])

        if self.return_dates:
            item = {
                "x": x,
                "y": y,
            }

            if self.y_dates is not None:
                item["y_dates"] = self.y_dates[idx]

            if self.pred_start_dates is not None:
                item["pred_start_date"] = self.pred_start_dates[idx]

            return item

        return x, y


def build_dataloader(
    root_dir="/mnt/sdc/tnchen/matchine_learning",
    split="train",
    task="90to90",
    batch_size=32,
    shuffle=None,
    num_workers=4,
    return_dates=False
):
    """
    构建 DataLoader。

    train 默认 shuffle=True
    eval 默认 shuffle=False
    """

    if shuffle is None:
        shuffle = True if split == "train" else False

    dataset = HouseholdPowerDataset(
        root_dir=root_dir,
        split=split,
        task=task,
        return_dates=return_dates
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataset, dataloader


if __name__ == "__main__":
    # 测试 90 -> 90
    train_dataset, train_loader = build_dataloader(
        split="train",
        task="90to90",
        batch_size=16
    )

    x, y = next(iter(train_loader))
    print("90to90 batch x:", x.shape)
    print("90to90 batch y:", y.shape)

    # 测试 90 -> 365
    train_dataset_365, train_loader_365 = build_dataloader(
        split="train",
        task="90to365",
        batch_size=16
    )

    x, y = next(iter(train_loader_365))
    print("90to365 batch x:", x.shape)
    print("90to365 batch y:", y.shape)
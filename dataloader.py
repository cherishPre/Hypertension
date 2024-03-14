import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TCMDataset(Dataset):
    def __init__(self, root_path="./data", modality=[], mode="train", seed=42):
        super().__init__()
        self.root_path = root_path
        self.modality = modality
        # 加载问诊数据
        
        self.data = pd.read_csv(Path(root_path, 'e2e.csv'), encoding="utf-8")
        # 加载面诊数据
        if 'face' in modality:
            self.face = pd.read_csv(Path(root_path, 'feature', f'point_face.csv'), encoding="utf-8", header=None)
            drop_list = []
            for i, v in self.data['姓名'].items():
                if v not in self.face[0].values:
                    drop_list.append(i)
            self.data = self.data.drop(drop_list)
        # 加载舌上数据
        if 'top' in modality:
            self.top = pd.read_csv(Path(root_path, 'feature', f'point_top.csv'), encoding="utf-8", header=None)
            drop_list = []
            for i, v in self.data['姓名'].items():
                if v not in self.top[0].values:
                    drop_list.append(i)
            self.data = self.data.drop(drop_list)
        # 加载舌下数据
        if 'bottom' in modality:
            self.bottom = pd.read_csv(Path(root_path, 'feature', f'ave_bottom.csv'), encoding="utf-8", header=None)
            drop_list = []
            for i, v in self.data['姓名'].items():
                if v not in self.bottom[0].values:
                    drop_list.append(i)
            self.data = self.data.drop(drop_list)
        if 'pulse' in modality:
            # 脉诊数据读文件
            self.pulse = pd.read_pickle(Path(root_path, 'feature', f'pulse.pkl'))
            drop_list = []
            for i, v in self.data['姓名'].items():
                if v not in self.pulse['file'].values:
                    drop_list.append(i)
            self.data = self.data.drop(drop_list)
        self.data = self.data.reset_index(drop=True)
        # normalize
        from sklearn.preprocessing import scale
        self.data.iloc[:, 2:-1] = scale(self.data.iloc[:, 2:-1])
        # 划分训练验证测试集
        train_idx, valtest_idx, _, _ = train_test_split(self.data.index, self.data[f'label'], test_size=0.4, random_state=seed, stratify=self.data[f'label'])
        val_idx, test_idx, _, _ = train_test_split(valtest_idx, self.data.loc[valtest_idx, f'label'], test_size=0.5, random_state=seed, stratify=self.data.loc[valtest_idx, f'label'])
        if mode == "train":
            self.data = self.data.loc[train_idx].reset_index(drop=True)
        elif mode == "val":
            self.data = self.data.loc[val_idx].reset_index(drop=True)
        elif mode == "test":
            self.data = self.data.loc[test_idx].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'id': self.data['ID'][index],
            'ask': torch.tensor(self.data.drop(columns=["ID", "姓名", "label"]).iloc[index].values, dtype=torch.float32),
            'label': torch.tensor(self.data[f"label"][index], dtype=torch.long)
        }
        if 'face' in self.modality:
            sample['face'] = torch.tensor(self.face[self.face[0] == self.data["姓名"][index]].drop(columns=[0]).iloc[0].values, dtype=torch.float32)
        if 'top' in self.modality:
            sample['top'] = torch.tensor(self.top[self.top[0] == self.data["姓名"][index]].drop(columns=[0]).iloc[0].values, dtype=torch.float32)
        if 'bottom' in self.modality:
            sample['bottom'] = torch.tensor(self.bottom[self.bottom[0] == self.data["姓名"][index]].drop(columns=[0]).iloc[0].values, dtype=torch.float32)
        if 'pulse' in self.modality:
            period_feature = torch.tensor(self.pulse[self.pulse["file"] == self.data["姓名"][index]].drop(columns=["file"]).iloc[0].period_feature, dtype=torch.float32)
            hemo_feature = torch.tensor(self.pulse[self.pulse["file"] == self.data["姓名"][index]].drop(columns=["file"]).iloc[0].hemo_feature, dtype=torch.float32)
            sample['pulse'] = torch.cat([period_feature, hemo_feature], dim=1)
        return sample


def TCMDataloader(root_path="./data", modality=[], seed=1, batch_size=512, num_workers=0):
    train_set = TCMDataset(root_path, modality, mode="train")
    val_set = TCMDataset(root_path, modality, mode="val")
    test_set = TCMDataset(root_path, modality, mode="test")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    modality_list = [
        [],
        ['face'],
        ['top'],
        ['bottom'],
        ['pulse'],
        ['face', 'top'],
        ['face', 'bottom'],
        ['face', 'pulse'],
        ['top', 'bottom'],
        ['top', 'pulse'],
        ['bottom', 'pulse'],
        ['face', 'top', 'bottom'],
        ['face', 'top', 'pulse'],
        ['face', 'bottom', 'pulse'],
        ['top', 'bottom', 'pulse'],
        ['face', 'top', 'bottom', 'pulse']
    ]
    for modality in modality_list:
        print(modality)
        train_loader, val_loader, test_loader = TCMDataloader(modality=modality)
        print(len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset))
    # train_loader, val_loader, test_loader = TCMDataloader(modality=['face', 'top', 'bottom', 'pulse'])
    # print(len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset))
    # print(len(train_loader.dataset))
    # print(len(val_loader.dataset))
    # print(len(test_loader.dataset))
    # for i, sample in enumerate(train_loader):
    #     print(sample)
    #     break
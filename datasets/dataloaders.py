import numpy as np
import pandas as pd
import nibabel as nib
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import monai.transforms as T

from monai.transforms import (
    Compose,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    RandRotate90,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    AddChanneld,
    ToTensor,
)


def read_nii(nii_path: str):
    nii = nib.load(nii_path)
    img = nii.get_fdata()

    return img


def new_split_data(
    data: pd.DataFrame,
    size=0.2,
    #    random_state=42
):
    train_data, val_data = train_test_split(
        data,
        test_size=size,
        # random_state=random_state,
        stratify=data["label2"],
    )
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    return train_data, val_data


def get_patch_from_img(self, img_H, img_L):
    # --------------------------------
    # randomly crop the patch
    # --------------------------------
    if self.n_channels == 3:
        H, W, _ = img_H.shape
    else:
        H, W = img_H.shape

    rnd_h = random.randint(0, max(0, H - self.image_size))
    rnd_w = random.randint(0, max(0, W - self.image_size))
    patch_H = img_H[:, rnd_h : rnd_h + self.image_size, rnd_w : rnd_w + self.image_size]
    patch_L = img_L[rnd_h : rnd_h + self.image_size, rnd_w : rnd_w + self.image_size]

    return patch_H, patch_L


class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None) -> None:
        super().__init__()
        self.df = data
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        case = self.df.iloc[index]
        id = str(case.ID).zfill(4)
        try:
            labels = case[["label1", "label2"]].to_numpy(np.float32)
        except:
            labels = None

        axial_img = self.transform(case.AxialPath)
        coronal_img = self.transform(case.CoronalPath)

        axial_img = axial_img.permute(0, 3, 1, 2)
        coronal_img = coronal_img.permute(0, 3, 1, 2)

        return {
            "axial_img": axial_img,
            "coronal_img": coronal_img,
            "labels": labels,
            "id": id,
        }


import numpy as np

NUM_SLICES = 16

TRAIN_TRANSFORM_IMG = Compose(
    [
        LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        Resize((256, 256, -1)),
        T.RandRotate90(),
        T.RandSpatialCrop((224, 224, -1), random_size=False),
        T.RandFlip(spatial_axis=0, prob=0.5),
        T.RandFlip(spatial_axis=1, prob=0.5),
        T.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
        ToTensor(),
    ]
)

VAL_TRANSFORM_IMG = Compose(
    [
        LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        Resize((256, 256, -1)),
        T.CenterSpatialCrop((224, 224, -1)),
        ToTensor(),
    ]
)

TEST_TRANSFORM_IMG = Compose(
    [
        LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
        ScaleIntensity(),
        Resize((256, 256, -1)),
        T.CenterSpatialCrop((224, 224, -1)),
        ToTensor(),
    ]
)

BASIC_TRANSFORMS = {
    "train": TRAIN_TRANSFORM_IMG,
    "val": VAL_TRANSFORM_IMG,
    "test": TEST_TRANSFORM_IMG,
}


def create_transforms(num_slices: int, resize=384, img_size=224):
    train_transform_img = Compose(
        [
            LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            Resize((resize, resize, num_slices)),
            T.RandRotate90(),
            T.RandSpatialCrop((img_size, img_size, num_slices), random_size=False),
            T.RandFlip(spatial_axis=0, prob=0.5),
            T.RandFlip(spatial_axis=1, prob=0.5),
            T.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
            ToTensor(),
        ]
    )

    val_transform_img = Compose(
        [
            LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            Resize((resize, resize, num_slices)),
            T.CenterSpatialCrop((img_size, img_size, num_slices)),
            ToTensor(),
        ]
    )

    test_transform_img = Compose(
        [
            LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            Resize((resize, resize, num_slices)),
            T.CenterSpatialCrop((img_size, img_size, num_slices)),
            ToTensor(),
        ]
    )

    transform_img = {
        "train": train_transform_img,
        "val": val_transform_img,
        "test": test_transform_img,
    }

    return transform_img


def my_dataloder(
    data: pd.DataFrame,
    val_data: pd.DataFrame = None,
    batch_size=16,
    num_workers=0,
    val_size=0.2,
    test=False,
    sample_list: list = None,
    #  seed=1234,
    transforms: dict = None,
):
    print("----Loading Dataset----")

    if transforms is None:
        transforms = BASIC_TRANSFORMS

    if test:
        test_dataset = MyDataset(data=data, transform=transforms["test"])
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        print("#### Test Data ####")
        print("# Test patients", len(test_dataset))
        print("-----------------------")

        return test_loader

    else:
        if val_data is None:
            train, val = new_split_data(
                data=data,
                size=val_size,
                # random_state=seed,
            )
            if sample_list is not None:
                labels = train[sample_list].to_numpy()
                pos_num = int(labels.sum(axis=1).sum())
                neg_num = labels.shape[0] - pos_num

                pos_weight = 1.0 / pos_num
                neg_weight = 1.0 / neg_num

                weights = [
                    neg_weight if labels.sum(axis=1)[i] == 0 else pos_weight
                    for i in range(labels.shape[0])
                ]
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                train_dataset = MyDataset(data=train, transform=transforms["train"])
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    sampler=sampler,
                    pin_memory=False,
                )
                val_dataset = MyDataset(data=val, transform=transforms["val"])
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=False,
                )
            else:
                train_dataset = MyDataset(data=train, transform=transforms["train"])
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=False,
                )
                val_dataset = MyDataset(data=val, transform=transforms["val"])
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=False,
                )

        else:
            train_dataset = MyDataset(data=data, transform=transforms["train"])
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            val_dataset = MyDataset(data=val_data, transform=transforms["test"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        print(
            "#### Dev Data ####",
        )
        print("# Train patinets: ", len(train_dataset))
        print("# Val patinets: ", len(val_dataset))
        print("-----------------------")

        return train_loader, val_loader

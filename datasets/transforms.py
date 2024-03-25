import torch
import numpy as np
import torch.nn.functional as F
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

def trans_tail(data, out_channels=32):
    # Reshape to (B, T, H, W, C)
    num_slice = data.shape[4]
    data = torch.transpose(data, 1, 4)

    # Trans T to `out_channels`
    conv = torch.nn.Conv3d(in_channels=num_slice, out_channels=out_channels, 
                           kernel_size=(1, 1, 1), stride=(1, 1, 1))
    data = F.relu(conv(data))

    # Reshape to (B, C, H, W, T)
    data = torch.transpose(data, 1, 4)

    return data

def resize_3d(data, size=224):
    """
    Shape (B, C, H, W, T) --> (B, C, size, size, T)
    """
    slince_num = data.shape[-1]
    
    resize_img = torch.nn.functional.interpolate(
        input=data.unsqueeze(0), size=(size, size, slince_num),
        )
    
    return resize_img.squeeze(0)


class Transforms:
    BASIC_TRANS = [
            LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True), 
            ScaleIntensity(),
        ]
    
    # def __init__(self, 
    #              ) -> None:
    #     # self.baisc_trans = [
    #     #     LoadImage(dtype=np.float32, image_only=True, ensure_channel_first=True), 
    #     #     ScaleIntensity(),
    #     # ]
        
    def create_train_trans(self, size=256, num_slices=32, add_trans:list=[]):
        trans = [
            Resize((size, size, num_slices)), 
            *add_trans, 
            ToTensor(),
            ]
        train_trans = self.BASIC_TRANS + trans
        
        return Compose(train_trans)

    def create_val_trans(self, size=256, num_slices=32):
        trans = [
            Resize((size, size, num_slices)), 
            ToTensor(),
            ]
        val_trans = self.BASIC_TRANS + trans
        
        return Compose(val_trans)
    
    def create_test_trans(self, size=256, num_slices=32):
        trans = [
            Resize((size, size, num_slices)), 
            ToTensor(),
            ]
        test_trans = self.BASIC_TRANS + trans
        
        return Compose(test_trans)
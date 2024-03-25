import os
import gc
import collections
import copy
import time
import torch
import wandb
from torch import nn
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import moveTo
# from utils.pytorchtools import EarlyStopping
# from Data.test_utils.tta import tta_inference
import copy
from datasets import my_dataloder
from torch.nn.parallel import DataParallel, DistributedDataParallel

def get_images_labels(data, plane, label, device):
    """
    Get images, labels pair according to the task
    
    Arg:
        plane: `coronal`, `axial`, `both`
        label: `lateral`, `medial`, `bilateral`
    
    Return:
        (images, labels)
    """
    
    if plane == "coronal":
        images = data["coronal_img"]
    elif plane == "axial":
        images = data["axial_img"]
    elif plane == "both":
        images = {"coronal_img": data["coronal_img"], 
                  "axial_img": data["axial_img"]}
    
    if label == "lateral":
        labels = data["labels"][:, 0]
    elif label == "medial":
        labels = data["labels"][:, 1]
    elif label == "bilateral":
        labels = data["labels"][:, :2]
    
    if len(labels.shape) == 1:
        labels = labels.view(labels.shape[0], -1)
    
    images = moveTo(images, device)
    labels = moveTo(labels, device)
    
    return images, labels
    

def train_one_epoch(model, criterion, optimizer, dataloader,
                    device="cuda", 
                    plane="both", 
                    label="bilateral"
                    ):
    """
    Args:
        plane: `coronal`, `axial`, `both`
        label: `lateral`, `medial`, `bilateral`
    """
    model.train()
    
    running_loss = []
    y_true = []
    y_pred = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images, labels = get_images_labels(data, plane, label, device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss.append(loss.item())
        
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = torch.sigmoid(outputs).detach().cpu().numpy()
            
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    
    # End training epoch
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    epoch_loss = np.mean(running_loss)
    
    gc.collect()

    return epoch_loss, y_true, y_pred


@torch.inference_mode()
def val_one_epoch(model, criterion, dataloader, 
                #   score_funcs:dict, 
                #   results:dict,
                  device="cuda", 
                  plane="both", 
                  label="bilateral"
                  ):
    """
    Args:
        coronal: `cornal`, `axial`, `both`
        label: `lateral`, `medial`, `bilateral`
    """
    model.eval()
    
    running_loss = []
    y_true = []
    y_pred = []
    

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images, labels = get_images_labels(data, plane, label, device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss.append(loss.item())

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = torch.sigmoid(outputs).detach().cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    
    # End training epoch
    
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    epoch_loss = np.mean(running_loss)
    
    gc.collect()

    return epoch_loss, y_true, y_pred 


@torch.inference_mode()
def test_one_epoch(model, dataloader, 
                  device="cuda", 
                  plane="both", 
                  label="bilateral"
                  ):
    """
    Args:
        coronal: `cornal`, `axial`, `both`
        label: `lateral`, `medial`, `bilateral`
    """
    model.eval()
    
    y_true = []
    y_pred = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images, labels = get_images_labels(data, plane, label, device)
        
        outputs = model(images)
        
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            y_hat = torch.sigmoid(outputs).detach().cpu().numpy()
            
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    gc.collect()

    return y_true, y_pred 

class Callback:
    def __init__(self, mode:str, 
                 metrics:dict=None, 
                 monitor:str=None, # save weight according to this metric
                 login_wandb=False, 
                 best_wts_num:int=None, 
                 ):
        self.metrics = {} if metrics is None else metrics
        
        if monitor in self.metrics:
            self.monitor = monitor
            self.best_metric = 0.5
        else:
            raise ValueError("Callback should be in metrics.")
            
        if mode == "train":
            self.train_results = collections.defaultdict(list)
            self.val_results = collections.defaultdict(list)
        elif mode == "test":
            self.test_results = collections.defaultdict(list)
            
        self.login_wandb = login_wandb

        if best_wts_num:
            self.best_wts = collections.deque(maxlen=best_wts_num)
            self.best_wt = None
        
    def on_train_begin(self):
        self.train_results["epoch"] = []
        self.val_results["epoch"] = []
        print("Start training ...")
    
    def on_epoch_begin(self, epoch:int):
        self.train_results["epoch"].append(epoch + 1)
        self.val_results["epoch"].append(epoch + 1)
    
    def on_train_epoch_end(self, epoch_loss_train, y_true_train, y_pred_train):
        self.train_results["train loss"].append(epoch_loss_train)
        for name, score_func in self.metrics.items():
            try:
                self.train_results[f"train {name}"].append(score_func(y_true_train, y_pred_train))
            except:
                self.train_results[f"train {name}"].append(float("NaN"))
        
        if self.login_wandb:
            try:
                roacus = self.train_results["train auc"][-1]
            except:
                roacus = None
            
            if isinstance(roacus, list):
                train_roacus = {}
                train_roacus["train auc_avg"] = round(np.mean(roacus), 3)
                for i, auc in enumerate(roacus, start=1):
                    train_roacus[f"train auc_{i}"] = auc
            elif roacus is not None:
                train_roacus = {}
                train_roacus["train auc"] = roacus
            else:
                train_roacus = {}
            
            train_log = {"train loss": epoch_loss_train}
            train_log.update(train_roacus)
                
            wandb.log(train_log, step=self.train_results["epoch"][-1])
        
    def on_val_epoch_end(self, epoch_loss_val, y_true_val, y_pred_val):
        self.val_results["val loss"].append(epoch_loss_val)
        for name, score_func in self.metrics.items():
            try:
                self.val_results[f"val {name}"].append(score_func(y_true_val, y_pred_val))
            except:
                self.val_results[f"val {name}"].append(float("NaN"))
        
        if self.login_wandb:
            try:
                roacus = self.val_results["val auc"][-1]
            except:
                roacus = None
            
            if isinstance(roacus, list):
                val_roacus = {}
                val_roacus["val auc_avg"] = round(np.mean(roacus), 3)
                for i, auc in enumerate(roacus, start=1):
                    val_roacus[f"val auc_{i}"] = auc
            elif roacus is not None:
                val_roacus = {}
                val_roacus["val auc"] = roacus
            else:
                val_roacus = {}

            val_log = {"val loss": epoch_loss_val}
            val_log.update(val_roacus)
            
            wandb.log(val_log, step=self.val_results["epoch"][-1])
                
    def update_best_wts(self, model:nn.Module, epoch):
        current_metric = self.val_results[f"val {self.monitor}"][-1]
        if isinstance(current_metric, list): 
            # For multi-label classification, the results is a list
            # Calculate the mean of metric
            current_metric = np.mean(current_metric)
        if self.best_metric < current_metric:
            print(f"Epoch {str(epoch+1).zfill(3)} {self.monitor} improved ({self.best_metric:.3f} --> {current_metric:.3f})")
            self.best_metric = current_metric
            self.best_wt = copy.deepcopy(model.state_dict())
            self.best_wts.append({
                f"E{epoch+1}_{self.monitor}[{self.best_metric:.3f}]": self.best_wt
                })
    
    def on_train_end(self, save_weights=True, checkpoints_dir=None):
        train_results = pd.DataFrame(self.train_results)
        val_results = pd.DataFrame(self.val_results)
        
        if save_weights and checkpoints_dir is not None:
            print("Save weights ...")
            for best_wt in self.best_wts:
                for k, v in best_wt.items():
                    wt_path = f"{checkpoints_dir}/{k}.pth"
                    torch.save(v, wt_path)
        
        return train_results, val_results


class Trainer:
    def __init__(self, 
                 model:nn.Module, 
                 desc:str, 
                 optimizer=None, 
                 criterion=None,
                 plane="both", 
                 label="bilateral", 
                 checkpoints_dir="./checkpoints", 
                 ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.plane = plane 
        self.label = label
        self.checkpoints_dir = f"{checkpoints_dir}/{desc}-{plane}-{label}"
        self.best_wt = None
        
    def train(self, 
              train_loader, 
              val_loader, 
              callback:Callback,
              epochs=10, 
              lr_scheduler=None,
              device="cuda", 
              save_weights=True,
              ):
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        start = time.time()
        
        self.model.to(device)
        callback.on_train_begin()
       
        for epoch in range(epochs):
            gc.collect()
            
            callback.on_epoch_begin(epoch=epoch)
            epoch_loss_train, y_true_train, y_pred_train = train_one_epoch(
                model=self.model, criterion=self.criterion, optimizer=self.optimizer, 
                dataloader=train_loader, device=device, plane=self.plane, label=self.label
                )
            callback.on_train_epoch_end(epoch_loss_train=epoch_loss_train, 
                                        y_true_train=y_true_train, 
                                        y_pred_train=y_pred_train)
            
            epoch_loss_val, y_true_val, y_pred_val = val_one_epoch(
                model=self.model, criterion=self.criterion, dataloader=val_loader, 
                device=device, plane=self.plane, label=self.label
                )

            callback.on_val_epoch_end(epoch_loss_val=epoch_loss_val, 
                                      y_true_val=y_true_val, 
                                      y_pred_val=y_pred_val)
            
            callback.update_best_wts(model=self.model, epoch=epoch)            

            msg = f"Epoch {str(epoch+1).zfill(3)} [Train loss: {epoch_loss_train:.5f}] [Val loss: {epoch_loss_val:.5f}]"
            print(msg)
            
            # The convention is to update the learning rate after every epoch
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(epoch_loss_val)
                else:
                    lr_scheduler.step()
        
        train_results, val_results = callback.on_train_end(
            save_weights=save_weights, checkpoints_dir=self.checkpoints_dir)
        self.best_wt = callback.best_wt
        
        return train_results, val_results
    
    def test(self, 
             test_loader, 
             metrics:dict, 
             device="cuda", 
             ):
        self.model.load_state_dict(self.best_wt)
        test_results = collections.defaultdict(list)
        
        y_true, y_pred = test_one_epoch(model=self.model, 
                                        dataloader=test_loader, 
                                        device=device, 
                                        plane=self.plane, 
                                        label=self.label)
        
        for name, score_func in metrics.items():
            metric = score_func(y_true, y_pred)
            test_results[f"test {name}"].append(metric)
            test_results[f"test avg_{name}"].append(round(np.mean(metric), 3))
        
        test_results = pd.DataFrame(test_results)
        return test_results
                
            
        
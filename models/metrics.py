import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, 
                             roc_curve, auc, )

def compute_aucs(y_true:list|np.ndarray, 
                 y_pred:list|np.ndarray, 
                 num_classes=4, 
                 decimal_places:int=None,
                 ):

    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    
    if num_classes == 1:
        auc = roc_auc_score(y_true, y_pred)
        return auc
    else:
        aurocs_lst = []
        for i in range(num_classes):
            auc = roc_auc_score(y_true[:, i].tolist(), y_pred[:, i].tolist())
            if decimal_places is not None:
                auc = float("{:.{}f}".format(auc, decimal_places))
                aurocs_lst.append(auc)
            else:
                aurocs_lst.append(auc)
        return aurocs_lst

def compute_acc(y_true:list|np.ndarray, 
                y_pred:list|np.ndarray, 
                num_classes=4, 
                threshold=0.5, 
                decimal_places:int=None,
                ):

    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    
    if num_classes == 1:
        predictions = np.where(y_pred >= 0.5, 1, 0)
        acc = accuracy_score(y_true, predictions)
        return acc
    else:
        acc_lst = []
        for i in range(num_classes):
            predictions = [1 if pred >= threshold else 0 for pred in y_pred[:, i].tolist()]
            labels = y_true[:, i].tolist()
            acc = accuracy_score(y_true=labels, y_pred=predictions)
            if decimal_places is not None:
                acc = float("{:.{}f}".format(acc, decimal_places))
                acc_lst.append(acc)
            else:
                acc_lst.append(acc)
        return acc_lst

def compute_se(y_true:list|np.ndarray, 
               y_pred:list|np.ndarray, 
               num_classes=4, 
               threshold=0.5, 
               decimal_places:int=None,
               ):

    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    
    if num_classes == 1:
        predictions = np.where(y_pred >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        sensitivity = tp / (tp + fn)
        return sensitivity
    else:
        se_lst = []
        for i in range(num_classes):
            predictions = [1 if pred >= threshold else 0 for pred in y_pred[:, i].tolist()]
            labels = y_true[:, i].tolist()
            
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            sensitivity = tp / (tp + fn)
            if decimal_places is not None:
                sensitivity = float("{:.{}f}".format(sensitivity, decimal_places))
                se_lst.append(sensitivity)
            else:
                se_lst.append(sensitivity)
        return se_lst


def compute_sp(y_true:list|np.ndarray, 
               y_pred:list|np.ndarray, 
               num_classes=4, 
               threshold=0.5, 
               decimal_places:int=None,
               ):

    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    
    if num_classes == 1:
        predictions = np.where(y_pred >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        specificity = tn / (tn + fp)
        return specificity
    else:
        sp_lst = []
        for i in range(num_classes):
            predictions = [1 if pred >= threshold else 0 for pred in y_pred[:, i].tolist()]
            labels = y_true[:, i].tolist()
            
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            specificity = tn / (tn + fp)
            if decimal_places is not None:
                specificity = float("{:.{}f}".format(specificity, decimal_places))
                sp_lst.append(specificity)
            else:
                sp_lst.append(specificity)
        return sp_lst


def find_best_threshold(y_true:list|np.ndarray, y_pred:list|np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # 计算离（0, 1）最近的点
    distance = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
    best_index = np.argmin(distance)

    best_threshold = thresholds[best_index]
    
    return best_threshold


import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    recall_score,
)
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def bina_metric(y_gt, y_pred, y_prob):

    def f1(y_gt, y_pred):
        """F1分数"""
        return f1_score(y_gt, y_pred, average='binary')

    def roc_auc(y_gt, y_prob):
        """ROC AUC分数"""
        return roc_auc_score(y_gt, y_prob)

    def precision_auc(y_gt, y_prob):
        """平均精确率 (Average Precision)"""
        return average_precision_score(y_gt, y_prob)

    def precision(y_gt, y_pred):
        """精确率"""
        return precision_score(y_gt, y_pred, zero_division=0)

    def recall(y_gt, y_pred):
        """召回率"""
        return recall_score(y_gt, y_pred, zero_division=0)
    

    
    precision = precision(y_gt, y_pred)
    recall = recall(y_gt, y_pred)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    roc_auc = roc_auc(y_gt, y_prob)
    accuracy = accuracy_score(y_gt, y_pred)
    # y_gt 真实0,1标签
    # y_pred 预测的阈值化的0,1 
    # y_prob 预测的概率
    return  precision, recall, f1, prauc, roc_auc,accuracy


def bina_metric2(y_gt, y_pred, y_prob):

    def f1(y_gt, y_pred):
        """F1分数"""
        return f1_score(y_gt, y_pred, average='binary')

    def roc_auc(y_gt, y_prob):
        """ROC AUC分数"""
        return roc_auc_score(y_gt, y_prob)

    def precision_auc(y_gt, y_prob):
        """平均精确率 (Average Precision)"""
        return average_precision_score(y_gt, y_prob)

    def precision(y_gt, y_pred):
        """精确率"""
        return precision_score(y_gt, y_pred, zero_division=0)

    def recall(y_gt, y_pred):
        """召回率"""
        return recall_score(y_gt, y_pred, zero_division=0)
    

    
    precision = precision(y_gt, y_pred)
    recall = recall(y_gt, y_pred)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    roc_auc = roc_auc(y_gt, y_prob)
    accuracy = accuracy_score(y_gt, y_pred)
    # y_gt 真实0,1标签
    # y_pred 预测的阈值化的0,1 
    # y_prob 预测的概率
    return  precision, recall, f1, prauc, roc_auc


def Adversarial_loss(domain_s, domain_t):
    # Initialize the pseudo-labels of the domain confrontation
    domain_labels = torch.cat((domain_s, domain_t), dim=0) 
    bs_s = domain_s.size(0)
    bs_t = domain_t.size(0)
    device = domain_s.device
    target_labels = torch.from_numpy(np.array([[0]]*bs_s + [[1]]*bs_t).astype('float32')).to(device)  # torch.Size([bs_s+bs_t, 1])，其中前面bs_s个为0，后面bs_t个为1
    # 上面的target_labels就是强硬指定source domain的标签为0，target domain的标签为1，也就是判别器要学会区分source domain和target domain
    domain_loss = nn.BCEWithLogitsLoss()(domain_labels, target_labels) 
    
    return domain_loss


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x / 1)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def instanceloss(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

#     loss = loss / (4 * d * d)
    return loss

def ortho_loss(source, target):
        n_source = source.size()[0]
        n_target = target.size()[0]
        if n_source < n_target:
            # 把source用自己的数据填充到和target一样多
            source = source.repeat(int(n_target / n_source) + 1, 1)[:n_target, :]
        elif n_target < n_source:
            target = target.repeat(int(n_source / n_target) + 1, 1)[:n_source, :]
        
        source_norm = torch.norm(source, p=2, dim=1, keepdim=True).detach()
        source_emb = source.div(source_norm.expand_as(source) + 1e-8)

        target_norm = torch.norm(target, p=2, dim=1, keepdim=True).detach()
        target_emb = target.div(target_norm.expand_as(target) + 1e-8)

        loss = torch.mean((source_emb.t().mm(target_emb)).pow(2))
        return loss

def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = input_data.device
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c   

########### 无用函数 ####################
def pearsonr(x, y):
    """Compute Pearson correlation.

    Args:
        x (torch.Tensor): 1D vector
        y (torch.Tensor): 1D vector of the same size as y.

    Raises:
        TypeError: not torch.Tensors.
        ValueError: not same shape or at least length 2.

    Returns:
        Pearson correlation coefficient.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError('Function expects torch Tensors.')

    if len(x.shape) > 1 or len(y.shape) > 1:
        raise ValueError(' x and y must be 1D Tensors.')

    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')

    if len(x) < 2:
        raise ValueError('x and y must have length at least 2.')

    # If an input is constant, the correlation coefficient is not defined.
    if bool((x == x[0]).all()) or bool((y == y[0]).all()):
        raise ValueError('Constant input, r is not defined.')

    mx = x - torch.mean(x)
    my = y - torch.mean(y)
    cost = (
        torch.sum(mx * my) /
        (torch.sqrt(torch.sum(mx**2)) * torch.sqrt(torch.sum(my**2)))
    )
    return torch.clamp(cost, min=-1.0, max=1.0)

def correlation_coefficient_loss(labels, predictions):
    """Compute loss based on Pearson correlation.

    Args:
        labels (torch.Tensor): reference values
        predictions (torch.Tensor): predicted values

    Returns:
        torch.Tensor: A loss that when minimized forces high squared correlation coefficient:
        \$1 - r(labels, predictions)^2\$  # noqa
    """
    return 1 - pearsonr(labels, predictions)**2

def r2_score(y_true, y_pred):
    """Compute R2 score.

    Args:
        y_true (torch.Tensor): reference values
        y_pred (torch.Tensor): predicted values

    Returns:
        torch.Tensor: R2 score
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def calculate_aucpr(y_true, y_pred):
    y_true = y_true.numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.numpy() if torch.is_tensor(y_pred) else y_pred

    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    aucpr = auc(recall, precision)
    return aucpr

###########################################



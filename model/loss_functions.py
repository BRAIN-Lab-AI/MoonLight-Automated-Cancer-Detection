import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Cross Entropy ----------
def cross_entropy():
    return nn.CrossEntropyLoss()


# ---------- 2. Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def focal():
    return FocalLoss()


# ---------- 3. Perceptual Loss (Softmax + One-hot + MSE) ----------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        inputs_soft = F.softmax(inputs, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        return self.mse(inputs_soft, targets_onehot)


def perceptual():
    return PerceptualLoss()


# ---------- 4. Composite Loss: CrossEntropy + Perceptual ----------
class CompositeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.percep = PerceptualLoss()

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)
        loss_percep = self.percep(inputs, targets)
        return self.alpha * loss_ce + self.beta * loss_percep


def composite():
    return CompositeLoss()


# ---------- 5. Dispatcher ----------
def get_loss_function(name):
    if name == "cross_entropy":
        return cross_entropy()
    elif name == "focal":
        return focal()
    elif name == "perceptual":
        return perceptual()
    elif name == "composite":
        return composite()
    else:
        raise ValueError(f"Unknown loss function: {name}")

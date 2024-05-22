import copy
import torch
import torch.nn as nn
from lightly.models.utils import batch_shuffle, batch_unshuffle, deactivate_requires_grad, update_momentum
from lightly.models.modules.heads import SimCLRProjectionHead
from torchmetrics.classification import Accuracy
from lightly.loss import NTXentLoss
from lightning import LightningModule
from torchvision.models import resnet18, resnet50
from lightly.models import ResNetGenerator
from utils import get_indices, l1_regularization
from torch.nn import CosineSimilarity
import numpy as np
import os

class NTXentLossPlus(NTXentLoss):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(**kwargs)
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

class SimCLRModel(LightningModule):
    def __init__(self, max_epochs, lr, momentum, weight_decay, args, negative_loss=False, enable_scheduler=True):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        if args.dataset not in ['cifar10', 'cifar100']:
            if args.backbone == 'resnet18':
                self.backbone = resnet18()
                feature_dim = 512
            elif args.backbone == 'resnet50':
                self.backbone = resnet50()
                feature_dim = 2048
            else:
                raise ValueError("Backbone not supported")
            self.backbone.fc = nn.Identity()
        else:
            if args.backbone == 'resnet18':
                resnet = ResNetGenerator("resnet-18")
                feature_dim = 512
            elif args.backbone == 'resnet50':
                resnet = ResNetGenerator("resnet-50")
                feature_dim = 2048
            else:
                raise ValueError("Backbone not supported")
            self.backbone = nn.Sequential(
                *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
            )
           
        self.projection_head = SimCLRProjectionHead(feature_dim, feature_dim, 128)

        self.mode = args.unlearn_mode
        self.alpha = args.alpha

        if self.mode == 'ng' or (self.alpha != 1 and self.alpha is not None):
            self.criterion = NTXentLossPlus(temperature=0.5, reduction='none')
        else:
            self.criterion = NTXentLossPlus(temperature=0.5)
        self.max_epochs=max_epochs
        self.lr = lr
        self.negative_loss = negative_loss
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.enable_scheduler = enable_scheduler
        _, retain_indices, unlearn_indices = get_indices(
            seed=args.seed, num_total_samples=args.num_total_samples, num_unlearn_samples=args.num_unlearn_samples)
        self.unlearn_indices = unlearn_indices
        self.beta = args.beta * len(unlearn_indices) / (len(retain_indices) + len(unlearn_indices))
        print(self.beta)
        self.similarity = CosineSimilarity(dim=1, eps=1e-6)  
        self.l1_reg = args.l1_reg

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        data, _, indices = batch
        unlearn_batch_indices = np.isin(indices.cpu().numpy(), self.unlearn_indices)
        x0 = data[0]
        x1 = data[1]
        z0 = self.forward(x0)
        z1 = self.forward(x1)            
        loss = self.criterion(z0, z1)

        if self.mode == 'ng':
            ubi = unlearn_batch_indices.repeat(2)
            rbi = (~unlearn_batch_indices).repeat(2)
            loss = (loss[rbi].sum() - loss[ubi].sum())/len(loss)

        if (self.alpha != 1 and self.alpha is not None):
            ubi = unlearn_batch_indices.repeat(2)
            rbi = (~unlearn_batch_indices).repeat(2)
            loss = (loss[rbi].sum() + self.alpha*loss[ubi].sum())/(rbi.sum()+self.alpha*ubi.sum())


        self.log("info_nce_loss", loss, on_step=True)
        if self.l1_reg is not None:
            l1_reg = l1_regularization(self)
            loss += self.l1_reg*l1_reg
            self.log("l1_regularization_term", l1_reg, on_step=True)
            
        if self.negative_loss:
            loss = -loss

        if len(data) == 3 and self.beta > 0 :
            x2 = data[2]
            z2 = self.forward(x2)
            a = z0[unlearn_batch_indices]
            b = z2[unlearn_batch_indices]
            loss_unlearn = self.similarity(a, b).mean()
            loss += self.beta*loss_unlearn
            self.log("unlearn_pos_sim_loss", loss_unlearn, on_step=True)
        self.log("train_loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        if not self.enable_scheduler:
            return [optim]
        return [optim], [scheduler]

class MocoModel(LightningModule):
    def __init__(self, max_epochs, lr, momentum, weight_decay, args, negative_loss=False, enable_scheduler=True):
        super().__init__()
        if args.dataset not in ['cifar10', 'cifar100']:
            self.backbone = resnet18()
            self.backbone.fc = nn.Identity()
        else:
            resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
            self.backbone = nn.Sequential(
                *list(resnet.children())[:-1],
                nn.AdaptiveAvgPool2d(1),
            )
        temperature = 0.1
        # create a moco model based on ResNet
        self.projection_head = nn.Linear(512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.mode = args.unlearn_mode
        self.alpha = args.alpha
        if self.mode == 'ng' or (self.alpha != 1 and self.alpha is not None):
            self.criterion = NTXentLossPlus(
                temperature=temperature, memory_bank_size=(4096, 128), reduction='none'
            )
        else:
            self.criterion = NTXentLossPlus(
                temperature=temperature, memory_bank_size=(4096, 128)
            )
        self.max_epochs=max_epochs
        self.lr = lr
        self.negative_loss = negative_loss
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.enable_scheduler = enable_scheduler

        _, retain_indices, unlearn_indices = get_indices(
                seed=args.seed, num_total_samples=args.num_total_samples, num_unlearn_samples=args.num_unlearn_samples)
        self.unlearn_indices = unlearn_indices
        self.beta = args.beta * len(unlearn_indices) / (len(retain_indices) + len(unlearn_indices))
        self.similarity = CosineSimilarity(dim=1, eps=1e-6)  
        self.l1_reg = args.l1_reg

    def training_step(self, batch, batch_idx):
        data, _, indices = batch
        x_q = data[0]
        x_k = data[1]
        # (x_q, x_k, x_k2), _, indices = batch

        unlearn_batch_indices = np.isin(indices.cpu().numpy(), self.unlearn_indices)

        # t_1(x), t_1(x'), t_2(x)
        # [t_1(x), t_1(x')], [t_1(x), t_2(x)]
        # anchor = t_1(x)
        
        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys k_t_1(x')
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        if self.mode == 'ng':
            loss = (loss[~unlearn_batch_indices].sum() - loss[unlearn_batch_indices].sum())/len(loss)

        if self.alpha != 1 and self.alpha is not None:
            ubi = unlearn_batch_indices
            rbi = (~unlearn_batch_indices)
            loss = (loss[rbi].sum() + self.alpha*loss[ubi].sum())/(rbi.sum()+self.alpha*ubi.sum())

        self.log("info_nce_loss", loss, on_step=True)  
        if self.l1_reg is not None:
            l1_reg = l1_regularization(self)
            loss += self.l1_reg*l1_reg
            self.log("l1_regularization_term", l1_reg, on_step=True)
        if self.negative_loss:
            loss = -loss

        if len(data) == 3 and self.beta > 0 :
            x_k2 = data[2]
            # get keys k_t_1(x') and k_t_2(x)
            k2, shuffle = batch_shuffle(x_k2)
            k2 = self.backbone_momentum(k2).flatten(start_dim=1)
            k2 = self.projection_head_momentum(k2)
            k2 = batch_unshuffle(k2, shuffle)
            # loss = self.criterion(q_t_1(x), k_t_1(x')) - cosinesim(q_t_1(x), k_t_2(x))
            if self.beta > 0:
                a = q[unlearn_batch_indices]
                b = k2[unlearn_batch_indices]
                loss_unlearn = self.similarity(a, b).mean()
                loss += self.beta*loss_unlearn
                self.log("unlearn_pos_sim_loss", loss_unlearn, on_step=True)
        self.log("train_loss_ssl", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        if not self.enable_scheduler:
            return [optim]
        return [optim], [scheduler]



class Classifier(LightningModule):
    def __init__(self, backbone, max_epochs, lr, momentum, num_classes, val_names=[], feature_dim=512):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(feature_dim, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.max_epochs = max_epochs
        self.val_metrics = nn.ModuleList([Accuracy(task="multiclass", num_classes=num_classes) for _ in val_names])

        self.lr = lr
        self.momentum = momentum
        self.val_names = val_names

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.val_metrics[dataloader_idx](y_hat, y)

    def on_validation_epoch_end(self):
        names = self.val_names
        for idx, metric in enumerate(self.val_metrics):
            self.log(f"{names[idx]}_acc", metric.compute(), on_epoch=True, prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.lr, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        scheduler = {'scheduler':torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs),
                     'name': 'fc_SGD_lr'}
        return [optim], [scheduler]

    def on_train_epoch_start(self):
        self.backbone.eval()

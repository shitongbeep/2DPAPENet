#!/usr/bin/env python
# encoding: utf-8
'''
@author: Shi Tong
@file: base_model.py
@time: 2022/11/9 14:12
'''
import torch
import os
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.criteria import MaskedMSELoss, SmoothL1Loss, Distance, FeatureDistance
from utils.vis_utils import save_depth_as_uint16png_upload, save_depth_as_uint8colored
from utils.metrics import AverageMeter
from utils.logger import logger
from typing import Dict, Any


class LightningBaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.depth_criterion = MaskedMSELoss()
        self.huberloss = SmoothL1Loss(20.)
        self.distance = Distance()
        self.mid_average_meter = AverageMeter()
        self.cd_average_meter = AverageMeter()
        self.dd_average_meter = AverageMeter()
        self.fuse_average_meter = AverageMeter()
        self.fuse_cd_average_meter = AverageMeter()
        self.refine_average_meter = AverageMeter()
        self.test_average_meter = AverageMeter()
        self.mylogger = logger(args)
        self.feature_distance = FeatureDistance()

    def configure_optimizers(self):
        # *optimizer
        if self.args.network_model == '_2dpaenet':
            # 训练没有CSPN++部分的backbone
            model_bone_params = [p for _, p in self.named_parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(model_bone_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(0.9, 0.99))
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            # 训练带有CSPN++的整个网络
            model_bone_params = [p for _, p in self.backbone.named_parameters() if p.requires_grad]
            model_new_params = [p for _, p in self.named_parameters() if p.requires_grad]
            model_new_params = list(set(model_new_params) - set(model_bone_params))
            optimizer = torch.optim.Adam([{
                'params': model_bone_params,
                'lr': self.args.learning_rate / 10.
            }, {
                'params': model_new_params
            }],
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay,
                                         betas=(0.9, 0.99))
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            # 固定backbone，训练CSPN++
            for p in self.backbone.parameters():
                p.requires_grad = False
            model_named_params = [p for _, p in self.named_parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(model_named_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay, betas=(0.9, 0.99))
        else:
            # 训练参数错误
            raise NotImplementedError('in base_model.py : optimizer wrong, config network_model and freeze_backbone in .yaml')

        # *lr_scheduler
        if self.args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=self.args.decay_step, gamma=self.args.decay_rate)
        elif self.args.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.args.decay_rate, patience=self.args.decay_step, verbose=True)
        elif self.args.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs - 4,
                eta_min=1e-5,
            )
        else:
            raise NotImplementedError('in base_model.py : lr_scheduler wrong, config lr_scheduler in .yaml')

        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'frequency': 1}

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, data, cd=True) -> Any:
        pass

    def training_step(self, data: Dict, batch_idx):
        if self.args.network_model == '_2dpaenet':
            if self.current_epoch < self.args.train_stage1:
                data = self.forward(data)
            else:
                data = self.forward(data, False)
            mid_branch_output = data['mid_branch_output']
            cd_branch_output = data['cd_branch_output']
            dd_branch_output = data['dd_branch_output']
            fuse_output = data['fuse_output']
            fuse_cd_output = data['fuse_cd_output']
            gt = data['gt']
            distillation_loss = data['distillation_loss']
            mid_loss = self.depth_criterion(mid_branch_output, gt)
            cd_loss = self.depth_criterion(cd_branch_output, gt)
            dd_loss = self.depth_criterion(dd_branch_output, gt)
            fuse_loss = self.depth_criterion(fuse_output, gt)
            fuse_cd_loss = self.depth_criterion(fuse_cd_output, gt)
            if self.current_epoch < self.args.train_stage0:
                loss = 1.2 * fuse_loss + 1.2 * fuse_cd_loss + 0.2 * (cd_loss + dd_loss + distillation_loss)
            elif self.current_epoch < self.args.train_stage1:
                loss = 1.8 * fuse_loss + 1.8 * fuse_cd_loss + 0.05 * (cd_loss + dd_loss + distillation_loss)
            else:
                loss = fuse_loss + 0.1 * (fuse_cd_loss + distillation_loss)
            # loss = fuse_loss + 0.1 * distance + 0.05 * cd_loss
            # loss = fuse_loss + 0.1 * distance
            self.log('train/loss', loss.item())
            loss_dict = {
                'cd_loss': cd_loss.item(),
                'mid_loss': mid_loss.item(),
                'dd_loss': dd_loss.item(),
                'fuse_loss': fuse_loss.item(),
                'fuse_cd_loss': fuse_cd_loss.item(),
                'distillation_loss': distillation_loss.item()
            }
            self.log('train/5_loss', loss_dict)
            return loss
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            fuse_output = data['fuse_output']
            gt = data['gt']
            loss = self.depth_criterion(refined_output, gt)
            fuse_loss = self.depth_criterion(fuse_output, gt)
            loss_dict = {'refined_loss': loss.item(), 'fuse_loss': fuse_loss.item()}
            self.log('train/loss', loss_dict)
            return loss
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            fuse_output = data['fuse_output']
            gt = data['gt']
            loss = self.depth_criterion(refined_output, gt)
            fuse_loss = self.depth_criterion(fuse_output, gt)
            loss_dict = {'refined_loss': loss.item(), 'fuse_loss': fuse_loss.item()}
            self.log('train/loss', loss_dict)
            return loss
        else:
            raise NotImplementedError('in base_model.py : train wrong, config network_model and freeze_backbone in .yaml')

    def validation_step(self, data, batch_idx):
        if self.args.network_model == '_2dpaenet':
            data = self.forward(data)
            mid_branch_output = data['mid_branch_output']
            cd_branch_output = data['cd_branch_output']
            dd_branch_output = data['dd_branch_output']
            fuse_output = data['fuse_output']
            fuse_cd_output = data['fuse_cd_output']
            gt = data['gt']

            self.mid_average_meter.update(mid_branch_output, gt)
            self.cd_average_meter.update(cd_branch_output, gt)
            self.dd_average_meter.update(dd_branch_output, gt)
            self.fuse_average_meter.update(fuse_output, gt)
            self.fuse_cd_average_meter.update(fuse_cd_output, gt)

            rmse_dict = {
                'mid_rmse': self.mid_average_meter.rmse,
                'cd_rmse': self.cd_average_meter.rmse,
                'dd_rmse': self.dd_average_meter.rmse,
                'fuse_rmse': self.fuse_average_meter.rmse,
                'fuse_cd_rmse': self.fuse_cd_average_meter.rmse,
            }
            self.log('val/4_rmse', rmse_dict, on_step=True)

            self.mylogger.conditional_save_img_comparison(batch_idx, data, fuse_output, self.current_epoch)
            return
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            fuse_output = data['fuse_output']
            gt = data['gt']
            self.refine_average_meter.update(refined_output, gt)
            self.fuse_average_meter.update(fuse_output, gt)
            loss = self.depth_criterion(refined_output, gt)
            fuse_loss = self.depth_criterion(fuse_output, gt)
            loss_dict = {'refined_loss': loss.item(), 'fuse_loss': fuse_loss.item()}
            self.log('val/2_loss', loss_dict, on_step=True)
            rmse_dict = {'refined_rmse': self.refine_average_meter.rmse, 'fuse_rmse': self.fuse_average_meter.rmse}
            self.log('val/2_rmse', rmse_dict, on_step=True)
            self.mylogger.conditional_save_img_comparison(batch_idx, data, refined_output, self.current_epoch)
            return loss
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            data = self.forward(data)
            refined_output = data['refined_depth']
            fuse_output = data['fuse_output']
            gt = data['gt']
            self.refine_average_meter.update(refined_output, gt)
            self.fuse_average_meter.update(fuse_output, gt)
            loss = self.depth_criterion(refined_output, gt)
            fuse_loss = self.depth_criterion(fuse_output, gt)
            loss_dict = {'refined_loss': loss.item(), 'fuse_loss': fuse_loss.item()}
            self.log('val/2_loss', loss_dict, on_step=True)
            rmse_dict = {'refined_rmse': self.refine_average_meter.rmse, 'fuse_rmse': self.fuse_average_meter.rmse}
            self.log('val/2_rmse', rmse_dict, on_step=True)
            self.mylogger.conditional_save_img_comparison(batch_idx, data, refined_output, self.current_epoch)
            return loss
        else:
            raise NotImplementedError('in base_model.py : validation wrong, config network_model and freeze_backbone in .yaml')

    def test_step(self, data, batch_idx):
        data = self.forward(data)
        if self.args.network_model == '_2dpaenet':
            pred = data['fuse_output']
        elif self.args.network_model == '_2dpapenet':
            pred = data['refined_depth']
        else:
            raise NotImplementedError('in base_model.py : test wrong, wrong network_model: ' + self.args.network_model)
        if self.args.mode == 'val':
            gt = data['gt']
            self.test_average_meter.update(pred, gt)
        else:
            str_i = str(self.current_epoch * self.args.batch_size + batch_idx)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(self.args.data_folder_save, path_i)
            save_depth_as_uint16png_upload(pred, path)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(self.args.data_folder_save + 'color/', path_i)
            save_depth_as_uint8colored(pred, path)

    def validation_epoch_end(self, outputs):
        if self.args.network_model == '_2dpaenet':
            self.mid_average_meter.compute()
            self.cd_average_meter.compute()
            self.dd_average_meter.compute()
            self.fuse_average_meter.compute()
            self.fuse_cd_average_meter.compute()
            self.log('val/rmse', self.fuse_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            # rmse_dict = {'mid_rmse': self.mid_average_meter.sum_rmse, 'cd_rmse': self.cd_average_meter.sum_rmse}
            # self.log('val/mid_cd_rmse', rmse_dict, on_epoch=True, sync_dist=True)
            self.mylogger.conditional_save_info(self.cd_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.mid_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.dd_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.fuse_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.fuse_cd_average_meter, self.current_epoch)
            if self.mid_average_meter.best_rmse > self.mid_average_meter.rmse:
                self.mylogger.save_img_comparison_as_best(self.current_epoch)
            self.mid_average_meter.reset_all()
            self.cd_average_meter.reset_all()
            self.dd_average_meter.reset_all()
            self.fuse_average_meter.reset_all()
            self.fuse_cd_average_meter.reset_all()
        elif self.args.network_model == '_2dpapenet' and self.args.freeze_backbone:
            self.refine_average_meter.compute()
            self.fuse_average_meter.compute()
            self.log('val/rmse', self.refine_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            self.mylogger.conditional_save_info(self.fuse_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch)
            if self.refine_average_meter.best_rmse > self.refine_average_meter.rmse:
                self.mylogger.save_img_comparison_as_best(self.current_epoch)
                self.mylogger.conditional_save_info(self.fuse_average_meter, self.current_epoch, True)
                self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch, True)
            self.refine_average_meter.reset_all()
            self.fuse_average_meter.reset_all()
        elif self.args.network_model == '_2dpapenet' and not self.args.freeze_backbone:
            self.refine_average_meter.compute()
            self.fuse_average_meter.compute()
            self.log('val/rmse', self.refine_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
            self.mylogger.conditional_save_info(self.fuse_average_meter, self.current_epoch)
            self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch)
            if self.refine_average_meter.best_rmse > self.refine_average_meter.rmse:
                self.mylogger.save_img_comparison_as_best(self.current_epoch)
                self.mylogger.conditional_save_info(self.fuse_average_meter, self.current_epoch, True)
                self.mylogger.conditional_save_info(self.refine_average_meter, self.current_epoch, True)
            self.refine_average_meter.reset_all()
            self.fuse_average_meter.reset_all()
        else:
            raise NotImplementedError('in base_model.py : validation end wrong, config network_model and freeze_backbone in .yaml')

    def test_epoch_end(self, outputs):
        if self.args.mode == 'val':
            self.test_average_meter.compute()
            self.mylogger.conditional_save_info(self.test_average_meter, self.current_epoch, False)

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

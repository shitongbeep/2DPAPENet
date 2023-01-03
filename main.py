import argparse
import importlib
import os

import pytorch_lightning as pl
import tensorboardX
import torch
import yaml
from dataloaders import kitti_loader
from easydict import EasyDict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
from torch.utils.data.dataloader import DataLoader


def getargs():
    parser = argparse.ArgumentParser(description="Sparse to Dense")
    parser.add_argument('-c', '--configs', default="./config/2DPAPENet-kitti.yaml")
    configs = parser.parse_args()
    with open(configs.configs, 'r') as config:
        args = yaml.safe_load(config)
    args.update(vars(configs))
    args = EasyDict(args)
    args.result = os.path.join('..', 'results')
    args.use_rgb = ('rgb' in args.input)
    args.use_d = 'd' in args.input
    args.use_g = 'g' in args.input
    print(args)
    return args


def build_loader(args):
    train_dataset_loader, val_dataset_loader, test_dataset_loader = None, None, None
    if not args.test:
        train_dataset = kitti_loader.KittiDepth('train', args)
        train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
        val_dataset = kitti_loader.KittiDepth('val', args)
        val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)
    else:
        if args.mode == 'val':
            val_dataset = kitti_loader.KittiDepth('val', args)
            val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)
        else:
            test_dataset = kitti_loader.KittiDepth('test_completion', args)
            test_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    return train_dataset_loader, val_dataset_loader, test_dataset_loader


def get_model(args):
    args.backbone_checkpoint = ''
    if args.network_model == '_2dpaenet':
        # model_file = importlib.import_module('network.' + args.network_model + "_large")
        model_file = importlib.import_module('network.' + args.network_model)
        args.checkpoint = 'best_2dpaenet'
    elif args.network_model == '_2dpapenet':
        model_file = importlib.import_module('network.' + args.network_model)
        args.baseline_only = True  # 只用学生网络
        args.checkpoint = 'best_2dpapenet'
        args.backbone_checkpoint = 'best_2dpaenet'
        args.freeze_backbone = False  # 暂时不考虑net和cspn一起训练，就分布训练
        args.monitor = 'val/rmse'
    else:
        raise ImportError("Error Import Model")
    return model_file.get_model(args)


if __name__ == "__main__":
    args = getargs()  # 加载参数
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    num_gpu = len(args.gpu)
    log_folder = '/root/tf-logs'
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name='kitti_depth_completion', default_hp_metric=False)
    profiler = SimpleProfiler(filename='profiler')
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
    pl.seed_everything(args.seed)

    train_dataset_loader, val_dataset_loader, test_dataset_loader = build_loader(args)  # 加载数据
    my_model = get_model(args)  # 加载模型
    # 保存和加载模型参数
    ckpt_path = None
    if os.path.isfile(args.checkpoint + '.ckpt') or os.path.isfile(args.backbone_checkpoint + '.ckpt'):
        print('load pre-trained model...')
        if args.test:
            # 加载模型测试
            my_model = my_model.load_from_checkpoint(args.checkpoint + '.ckpt', args=args, strict=False)
        elif args.network_model == '_2dpapenet' and not args.freeze_backbone:
            # 继续训练
            # my_model = my_model.load_from_checkpoint(args.checkpoint + '.ckpt', args=args)
            my_model.backbone = my_model.backbone.load_from_checkpoint(args.backbone_checkpoint + '.ckpt', args=args, strict=False)
        elif args.network_model == '_2dpapenet' and args.freeze_backbone:
            # 只加载backbone，训练cspn
            my_model.backbone = my_model.backbone.load_from_checkpoint(args.backbone_checkpoint + '.ckpt', args=args, strict=False)
        else:
            my_model = my_model.load_from_checkpoint(args.checkpoint + '.ckpt', args=args, strict=False)
    checkpoint_callback = ModelCheckpoint(
        dirpath='.',
        filename=args.checkpoint,
        monitor=args.monitor,
        mode='min',
        save_last=True,
    )
    # 是否要SWA
    if args.SWA:
        swa = [
            StochasticWeightAveraging(swa_lrs=args.swa_lr, swa_epoch_start=args.swa_epoch_start, annealing_epochs=1, device=torch.device('cuda', 0))
        ]
    else:
        swa = []
    # 开始训练
    if not args.test:
        print('Start training...')
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[i for i in range(num_gpu)],
            strategy='ddp',
            max_epochs=args.epochs,
            limit_train_batches=10,
            limit_val_batches=10,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step'),
                EarlyStopping(monitor=args.monitor, patience=args.stop_patience, mode='min', verbose=True),
            ] + swa,
            logger=tb_logger,
            profiler=profiler,
            log_every_n_steps=20,
            # gradient_clip_algorithm='norm',
            # gradient_clip_val=1
            )
        trainer.fit(my_model, train_dataset_loader, val_dataset_loader)
    else:
        if not os.path.exists(args.data_folder_save):
            os.mkdir(args.data_folder_save)
        print('Start testing...')
        assert num_gpu == 1, 'only support single GPU testing!'
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[i for i in range(num_gpu)],
            strategy='ddp',
            #  limit_test_batches=10,
            logger=tb_logger,
            profiler=profiler)
        if args.mode == 'val':
            trainer.test(my_model, val_dataset_loader)
        else:
            trainer.test(my_model, test_dataset_loader)

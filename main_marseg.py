import argparse
import copy
import datetime
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch_ema import ExponentialMovingAverage
from yacs.config import CfgNode as CN

from dataloader import SegmentationDataset
from engine_marseg import train_one_epoch, validate
from models import marseg
from models.mar import mar_layers
from models.marseg import MARSeg
from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder
from util.loss import DiceLoss


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_args_parser():
    parser = argparse.ArgumentParser('MAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='mar_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--n_frames', default=1, type=int, help='Number of adjacent slices to use')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="./pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=6)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=4)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/msd_dataset', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=3, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use.')

    # caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--checkpoint_name', default='checkpoint-last.pth', help='checkpoint file name')
    #pretrained models
    parser.add_argument('--mar_checkpoint_path', type=str, default='.pretrained_mar/checkpoint-epoch_180.pth', help='Path to MAR model checkpoint')

    # Fusion module params
    parser.add_argument('--layer_indices', nargs='+', type=int, default=[8, 9, 10, 11], help="last 4 layers")

    return parser


def initialize_mar_model(args, device, checkpoint_path=None):

    mar_model = mar_layers.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )
    mar_model.to(device)
    trainable_params = list(mar_model.parameters())

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        checkpoint_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        desired_prefixes = ['encoder_pos_embed_learned', 'z_proj.', 'encoder_blocks.', 'decoder.', 'quant_conv.']

        filtered_model_dict = {k: v for k, v in checkpoint_state_dict.items() 
                               if any(k.startswith(prefix) for prefix in desired_prefixes)}

        model_state_dict = mar_model.state_dict()

        model_state_dict.update(filtered_model_dict)

        mar_model.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded MAR checkpoint from {checkpoint_path}")
    else:
        print("No MAR checkpoint found or checkpoint path invalid. Using randomly initialized MAR model.")

    for param in mar_model.encoder_blocks.parameters():
        param.requires_grad = False
    print("MAR model encoder parameters frozen.")

    return mar_model
    

def initialize_segmentation_model(mar_model, pretrained_vae_path, num_classes=3, resolution=224):
    model = MARSeg(mar_model,
                    pretrained_vae_path,
                    embed_dim=16,
                    ch_mult=(1, 1, 2, 2, 4),
                    num_classes=num_classes,
                    resolution=256,
                    layer_indices=tuple(args.layer_indices),
                    fuse_dim=256,
                    reduction_ratio=16,
                    pool_types=['avg','max']
                    )
    return model


def initialize_training(model, device, num_classes=3, learning_rate=1e-4):
    if learning_rate is None:
        learning_rate = 1e-4
    class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=None)
    dice_loss = DiceLoss(n_classes=num_classes)
    def combined_loss(logits, targets):
        ce   = ce_loss(logits, targets)
        dice = dice_loss(logits, targets, softmax=True)
        return 2.0 * ce + 1.0 * dice
    criterion = combined_loss
    decoder_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decoder_parameters, lr=learning_rate, betas=(0.9, 0.999))
    return criterion, optimizer


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # train dataset
    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = SegmentationDataset(args.data_path, split='train', num_slices=args.n_frames)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # validation dataset
    dataset_val = SegmentationDataset(args.data_path, split='val', num_slices=args.n_frames)
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # test dataset
    dataset_test = SegmentationDataset(args.data_path, split='test', num_slices=args.n_frames)
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # Load the Model
    mar_model = initialize_mar_model(args, device ,args.mar_checkpoint_path)
    model = initialize_segmentation_model(mar_model, args.vae_path, args.class_num, args.img_size)
    model.to(device)

    criterion, optimizer = initialize_training(model, device, args.class_num, args.lr)

    if args.resume and os.path.exists(os.path.join(args.resume, args.checkpoint_name)): 
        checkpoint_path = os.path.join(args.resume, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if 'loss_scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        
        print(f"Resumed training from {checkpoint_path} at epoch {start_epoch}")
    else:
        print("Training from scratch")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model


    # Evaluate the model on test data if --evaluate is set
    if args.evaluate:
        torch.cuda.empty_cache()
        avg_loss, avg_dice, avg_iou = validate(
            model = model,
            args=args,
            epoch=args.start_epoch,
            data_loader=data_loader_test,
            criterion = criterion,
            device = device,
            log_writer=log_writer,
        ) 
        print("Evaluation completed.")
        return


    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    loss_scaler = GradScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_val_loss = float('inf')
    best_val_dice = float('-inf')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            data_loader_train,
            criterion,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if (epoch + 1) % 1 == 0:
            val_loss, val_dice, val_iou = validate(
                model,
                data_loader_val,
                criterion,
                device,
                epoch,
                log_writer=log_writer,
                args=args
            )

            if isinstance(val_dice, dict):
                val_dice = np.mean(list(val_dice.values()))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best_val_loss=best_val_loss, ema_params=None,
                                epoch_name=f"best_loss_{epoch}")
                print(f"New best loss model saved at epoch {epoch}")
            elif val_dice > best_val_dice:
                best_val_dice = val_dice
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, best_val_dice=best_val_dice, ema_params=None,
                                epoch_name=f"best_dice_{epoch}")
                print(f"New best dice model saved at epoch {epoch}")

        # save checkpoint
        if epoch % args.save_last_freq == 0:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            ema_params=None, best_val_loss=best_val_loss, epoch_name=f"{epoch}")

        if epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            ema_params=None, best_val_loss=best_val_loss, epoch_name="last")

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    epoch_last = args.epochs
    
    test_loss, test_dice, test_iou = validate(
            model,
            data_loader_test,
            criterion,
            device,
            epoch_last,
            log_writer=log_writer,
            args=args
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.log_dir = args.output_dir
    main(args)

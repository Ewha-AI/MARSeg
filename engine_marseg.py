import copy
import json
import math
import os
import random
import shutil
import sys
import time
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_fidelity
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from skimage import measure
from torch.cuda.amp import autocast

import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import MetricLogger


def prepare_mask_for_overlay(mask):
    pancreas_mask = (mask == 1)
    tumor_mask = (mask == 2)
    return np.stack([pancreas_mask, tumor_mask], axis=-1)    


def train_one_epoch(model,
                   data_loader: Iterable,
                   criterion,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   epoch: int,
                   loss_scaler: torch.cuda.amp.GradScaler,
                   log_writer=None,
                   args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, class_label, original_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        class_label = class_label.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(samples, class_label)
            loss = criterion(outputs, targets)

        loss_scaler.scale(loss).backward()

        loss_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        loss_scaler.step(optimizer)
        loss_scaler.update()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model,
             data_loader,
             criterion,
             device,
             epoch,
             log_writer=None,
             args=None):
    model.eval()

    metric_logger = MetricLogger()
    header = f'Validation: Epoch [{epoch}]'
    print_freq = 20
    num_classes = args.class_num
    total_loss = 0.0

    output_dir_lower = args.output_dir.lower() if args and hasattr(args, 'output_dir') else ""
    tp = torch.zeros(num_classes).to(device)
    fp = torch.zeros(num_classes).to(device)
    fn = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for data_iter_step, (samples, targets, class_label, original_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            class_label = class_label.to(device, non_blocking=True)
            original_masks = original_mask.to(device, non_blocking=True)

            with autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
            loss_value = loss.item()
            total_loss += loss_value

            final_preds  = torch.argmax(outputs, dim=1)
            batch_size = final_preds.shape[0]

            for b in range(batch_size):
                pred_b = final_preds[b]
                target_b = original_masks[b]
                for cls in range(num_classes):
                    pred_cls = (pred_b == cls).float()
                    target_cls = (target_b == cls).float()
                    tp[cls] += (pred_cls * target_cls).sum()
                    fp[cls] += (pred_cls * (1 - target_cls)).sum()
                    fn[cls] += ((1 - pred_cls) * target_cls).sum()

    dsc_per_class = {}
    iou_per_class = {}

    for cls in range(num_classes):
        if tp[cls] + fp[cls] + fn[cls] == 0:
            dsc = 1.0
            iou = 1.0
        else:
            dsc = (2 * tp[cls] + 1e-6) / (2 * tp[cls] + fp[cls] + fn[cls] + 1e-6)
            iou = (tp[cls] + 1e-6) / (tp[cls] + fp[cls] + fn[cls] + 1e-6)
        dsc_per_class[cls] = dsc.item()
        iou_per_class[cls] = iou.item()

    mean_dsc = np.mean(list(dsc_per_class.values()))
    mean_iou = np.mean(list(iou_per_class.values()))

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    if log_writer is not None:
        log_writer.add_scalar('val_loss', avg_loss, epoch)
        for cls, dsc_score in dsc_per_class.items():
            log_writer.add_scalar(f'val_dice_class_{cls}', dsc_score, epoch)
        for cls, iou_score in iou_per_class.items():
            log_writer.add_scalar(f'val_iou_class_{cls}', iou_score, epoch)
        log_writer.add_scalar('val_dice_mean', mean_dsc, epoch)
        log_writer.add_scalar('val_iou_mean', mean_iou, epoch)

    print(f"Validation Loss: {avg_loss:.4f}")
    for cls, (dsc_score, iou_score) in enumerate(zip(dsc_per_class.values(), iou_per_class.values())):
        print(f"Class {cls} Dice: {dsc_score:.4f}, IoU: {iou_score:.4f}")
    print(f"Mean Dice: {mean_dsc:.4f}, Mean IoU: {mean_iou:.4f}")

    if misc.get_rank() == 0:
        metrics = {
            "epoch": epoch,
            "val_loss": avg_loss,
            "val_dice_mean": mean_dsc,
            "val_iou_mean": mean_iou,
            "val_dice_per_class": dsc_per_class,
            "val_iou_per_class": iou_per_class
        }

        json_path = os.path.join(args.output_dir, 'validation_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        existing_data.append(metrics)
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
        print(f"Validation metrics saved to {json_path}")

    return avg_loss, dsc_per_class, iou_per_class


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return

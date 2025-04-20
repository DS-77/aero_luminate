"""
This module is the training script for the Aero-Luminate pipeline.

Author: Deja S.
Version: 1.0.1
Edited: 19-04-2025
"""

import os
import tqdm
import json
import torch
import argparse
from torch import Tensor
from datetime import date
from torch.optim import Adam
from torch.optim import AdamW
import torch.nn.functional as F
import torchvision.models as models
from configs.config import load_config
from utils.dataset import AISD_dataset
from torch.utils.data import DataLoader
from models import shadow_diff, adapter_shadow
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def combined_loss_fn(original_img, shadow_mask_pred, ground_truth_mask, shadow_remove_img) -> Tensor:
    lambda_1 = 1.0
    lambda_2 = 0.5
    lambda_3 = 0.1

    # Binary Cross Entropy Loss
    bce_loss = F.binary_cross_entropy(shadow_mask_pred, ground_truth_mask)

    # L1 Loss
    l1_loss = F.l1_loss(shadow_remove_img, original_img)

    # Perceptual Loss
    vgg = models.vgg19(pretrained=True).features
    vgg = vgg.to(shadow_remove_img.device).eval()
    with torch.no_grad():
        original_feat = vgg(original_img)
        reconstructed_feat = vgg(shadow_remove_img)
    perceptual_loss = F.mse_loss(original_feat, reconstructed_feat)

    total_loss = (lambda_1 * bce_loss) + (lambda_2 * l1_loss) + (lambda_3 * perceptual_loss)

    return total_loss


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", required=True, help="Path to model configuration (conf.yaml) file.")
    parser.add_argument("-o", "--out_dir", required=False, help="Output directory path.")
    opts = parser.parse_args()

    conf_file = opts.conf

    # Check if config file path exist
    if not os.path.exists(conf_file):
        print(f"ERROR: '{conf_file}' can not be found.")
        exit()

    # Read configuration file
    configs = load_config(conf_file)

    # Training data path
    train_img_path = configs["model_config"]["train_img_path"]
    train_mask_path = configs["model_config"]["train_mask_path"]
    valid_img_path = configs["model_config"]["valid_img_path"]
    valid_mask_path = configs["model_config"]["valid_mask_path"]

    if not os.path.exists(train_img_path):
        print(f"ERROR: Training image path does not exist: '{train_img_path}'")
        exit()

    if not os.path.exists(train_mask_path):
        print(f"ERROR: Training image path does not exist: '{train_mask_path}'")
        exit()

    if not os.path.exists(valid_img_path):
        print(f"ERROR: Validation image path does not exist: '{valid_img_path}'")
        exit()

    if not os.path.exists(valid_mask_path):
        print(f"ERROR: Validation mask path does not exist: '{valid_mask_path}'")
        exit()

    # Create required directories
    output_path_root = opts.out_dir if opts.out_dir is not None else configs["model_config"]["output_path"]
    temp_output_path = os.path.join(output_path_root, "train")
    output_path = os.path.join(temp_output_path, f"exp_{date.today()}")
    weights_path = os.path.join(output_path, "weights")
    log_path = os.path.join(output_path, "logs")

    # Root Dir Path
    if not os.path.exists(output_path_root):
        os.mkdir(output_path_root)

    # Train Dir Path
    if not os.path.exists(temp_output_path):
        os.mkdir(temp_output_path)

    # Experiment Dir Path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Log Dir Path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Weights Dir Path
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    # Load dataset
    train_dataset = AISD_dataset(train_img_path, train_mask_path)
    valid_dataset = AISD_dataset(valid_img_path, valid_mask_path)

    train_dataloader = DataLoader(train_dataset, batch_size=configs["model_config"]["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # print(f"Train set: {train_dataset.__len__()}")
    # print(f"Valid set: {valid_dataset.__len__()}")
    # exit()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialise the models

    # Adapter Shadow Configs
    adapter_args = argparse.Namespace()
    adapter_args.vpt = True
    adapter_args.all = True
    adapter_args.net = "sam"
    adapter_args.lora = True
    adapter_args.backbone = "b5"
    adapter_args.ratio_bt = 1.0
    adapter_args.mb_ratio = 0.25
    adapter_args.small_size = True
    adapter_args.multi_branch = True
    adapter_args.skip_adapter = True
    adapter_args.plug_image_adapter = True

    # Shadow Diffusion Configs
    shadow_diff_config_path = configs["shadow_diff"]["config_json"]
    with open(shadow_diff_config_path, 'r') as f:
        shadow_diff_opt = json.load(f)

    # Adapter Shadow Model
    adapter_shadow = adapter_shadow.adapt_sam_model_registry('vit_b', None, checkpoint=configs["adapter_shadow"]["pre_train_weights_path"])

    # Shadow Diffusion Model
    shadow_diff = shadow_diff.create_model(shadow_diff_opt)
    shadow_diff.set_new_noise_schedule(
        shadow_diff_opt['model']['beta_schedule']['train'],
        schedule_phase='train'
    )

    print(f"Device: {device}")
    adapter_shadow = adapter_shadow.to(device)
    shadow_diff = shadow_diff.to(device)

    optimiser = Adam(list (adapter_shadow.parameters()) + list (shadow_diff.parameters()), lr=float(configs["model_config"]["lr"]))
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=3, factor=0.5)

    # Run training loop
    num_epochs = configs["model_config"]["epochs"]
    best_loss = float("inf")

    # TensorBoard monitoring
    writer = SummaryWriter(log_dir=log_path)

    for epoch in tqdm.tqdm(range(num_epochs)):
        for batch_index, (imgs, masks) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Predict shadow mask
            pred_shadow_masks = adapter_shadow(imgs)

            # Remove shadow
            no_shadow_imgs = shadow_diff(pred_shadow_masks, imgs)

            # Compute the loss
            loss = combined_loss_fn(imgs, pred_shadow_masks, masks, no_shadow_imgs)

            # Backprop
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Log to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_index)

            # Save models every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(adapter_shadow.state_dict(), os.path.join(weights_path, f"checkpoint_{epoch + 1}_adapt_shadow.pth"))
                torch.save(shadow_diff.state_dict(), os.path.join(weights_path, f"checkpoint_{epoch + 1}_shadow_diff.pth"))

            # Save best models
            if loss < best_loss:
                best_loss = loss
                torch.save(adapter_shadow.state_dict(),
                           os.path.join(weights_path, f"best_checkpoint_adapt_shadow.pth"))
                torch.save(shadow_diff.state_dict(),
                           os.path.join(weights_path, f"best_checkpoint_shadow_diff.pth"))

    writer.close()
"""
This module is the training script for the Shadow-detect pipeline.

Author: Deja S.
Version: 1.0.0
Edited: 12-04-2025
"""

import os
import tqdm
import torch
import logging
import argparse
from datetime import date

from torch import Tensor
from torch.optim import Adam
from torch.optim import AdamW
import torch.nn.functional as F
import torchvision.models as models
from config.config import load_config
from torch.utils.data import DataLoader
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
    training_img_path = configs["model_config"]["training_img_path"]
    training_mask_path = configs["model_config"]["training_mask_path"]

    if not os.path.exists(training_img_path):
        print(f"ERROR: Training image path does not exist: '{training_img_path}'")
        exit()

    if not os.path.exists(training_mask_path):
        print(f"ERROR: Training image path does not exist: '{training_mask_path}'")
        exit()

    # Create required directories
    output_path_root = opts.out_dir if opts.out_dir is not None else configs["model_config"]["output_path"]
    output_path = os.path.join(output_path_root, "train")
    weights_path = os.path.join(output_path, "weights")
    log_path = os.path.join(output_path, "logs")

    # Root Dir Path
    if not os.path.exists(output_path_root):
        os.mkdir(output_path_root)

    # Train Dir Path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Log Dir Path
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # Weights Dir Path
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    # Load dataset

    # Device selection
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Initialise the models
    adapter_shadow = None
    shadow_diff = None

    optimiser = Adam(list (adapter_shadow.parameters()) + list (shadow_diff.parameters()), lr=configs["model_config"]["lr"])
    scheduler = ReduceLROnPlateau(optimiser, mode='min', patience=3, factor=0.5)

    # Run training loop

        # Save best models
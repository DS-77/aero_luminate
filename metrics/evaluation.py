"""
This module evaluates the testing image(s) for the Aero_Illuminate project using the following metrics: Shadow Recovery
Index (SRI), Colour Dissimilarity (CD), and Gradient Magnitude Similarity Derivation (GMSD). This evaluation script is
based on the implementation described by Mingqiang Guo et al. "Shadow removal method for high-resolution aerial remote
sensing images based on region group matching". We also included standard inpainting metrics: Structural Similarity
Index Measure (SSIM), Signal-to-Noise Ratio (PSNR), Mean Square Error (MSE) and Root Mean Square Error(RMSE).

Author: Deja S.
Created: 27-03-2025
Edited: 04-04-2025
Version: 1.0.2
"""

import os
import tqdm 
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import date
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim_fn


def convert_to_binary(image, threshold=127):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    _, binary_image = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)
    return binary_image

def MSE(shadow_img, no_shadow_img) -> float:
    # Lower MSE indicates better results
    return np.mean((shadow_img - no_shadow_img) ** 2)

def RMSE(shadow_img, no_shadow_img) -> float:
    # Lower RMSE means better results
    return sqrt(MSE(shadow_img, no_shadow_img))

def PSNR(shadow_img, no_shadow_img) -> float:
    # Higher PSNR means better results
    mse = MSE(shadow_img, no_shadow_img)

    if mse == 0:
        return 100.0

    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))


def SRI(img) -> float:
    # Lower SRI means less shadows are present
    # K -> number of channels
    # N -> the total number of pixels selected from the same area type 
    # i -> represent the current pixel
    # F -> represent the pixel value after shadow removal
    # 𝜇 -> average pixel value from no-shadow region
    
    # Convert to greyscale
    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Compute the local mean intensity
    local_mean = cv.blur(grey_img, (5, 5))
    
    # Compute the local variance
    local_var = cv.blur(grey_img ** 2, (5,5)) - local_mean ** 2
    local_var = np.maximum(local_var, 1e-10)
    
    # Compute local coefficient of variance
    co_var = np.sqrt(local_var) / local_mean
    
    return 1 - np.mean(co_var)

def compute_SRI(shadow_img, no_shadow_img) -> tuple:
    
    original_sri = SRI(shadow_img)
    processed_sri = SRI(no_shadow_img)
    
    return original_sri, processed_sri
    

def CD(shadow_img, no_shadow_img) -> float:
    # Lower value is better
    # 𝛥R -> degree of colour difference for red channel
    # 𝛥G -> degree of colour difference for green channel
    # 𝛥B -> degree of colour difference for blue channel
    
    # Split the image channels
    B1, G1, R1 = cv.split(shadow_img)
    B2, G2, R2 = cv.split(no_shadow_img)
    
    # Compute the colour difference
    delta_R = np.abs(R1 - R2)
    delta_G = np.abs(G1 - G2)
    delta_B = np.abs(B1 - B2)
    
    # Compute the CD
    cd = np.mean(np.abs(delta_R + delta_G + delta_B) / 3)
    
    return cd
    

def GM(img, kernel_x, kernel_y) -> np.array:
    # Gradient Magnitude
    
    # Compute the gradients for x and y
    grad_x = cv.filter2D(img.astype(float), -1, kernel_x)
    grad_y = cv.filter2D(img.astype(float), -1, kernel_y)
    
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return grad_magnitude
    

def GMS (mrsd, mnsd, c=0.0026) -> float:
    # Gradient Magnitude Similarity
    
    top = 2 * mrsd * mnsd + c
    bot = mrsd ** 2 + mnsd ** 2 + c
    
    return top / bot


def GMSD(shadow_img, no_shadow_img) -> float:
    # Lower value is better
    
    # Change images to grey
    grey_shadow_img = cv.cvtColor(shadow_img, cv.COLOR_BGR2GRAY)
    grey_no_shadow_img = cv.cvtColor(no_shadow_img, cv.COLOR_BGR2GRAY)
    
    # Sobel filters
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])
    
    # compute gradient
    mrsd = GM(grey_shadow_img, kernel_x, kernel_y)
    mnsd = GM(grey_no_shadow_img, kernel_x, kernel_y)
    
    gms_value = GMS(mrsd, mnsd)
    
    gmsm = np.mean(gms_value)
    
    gmsd = np.sqrt(np.mean((gms_value - gmsm)**2))
    
    return gmsd

def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, type=str, help="The processing mode for images: 'directory' or 'single'")
    parser.add_argument("-i", "--input_shadow", required=True, type=str, help="The path to the input directory or image with a shadow.")
    parser.add_argument("-n", "--input_no_shadow", required=True, type=str, help="The path to the input directory or image without shadow.")
    parser.add_argument("-o", "--output_path", required=False, default="./runs/eval", help="The path to the output directory.")
    opts = parser.parse_args()
    
    # Required Variables
    input_shadow = opts.input_shadow
    input_no_shadow = opts.input_no_shadow
    output_dir = opts.output_path
    mode = opts.mode
    
    # Check if the files exist
    if not os.path.exists(input_shadow):
        print(f"ERROR: {input_shadow} path does not exists!")
        exit()
        
    if not os.path.exists(input_no_shadow):
        print(f"ERROR: {input_no_shadow} path does not exist!")
        exit()
            
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
        
    # Single image case    
    if mode == "single":
        
        # All metrics
        cd = 0
        sri_1 = 0
        sri_2 = 0
        mse = 0
        gmsd = 0
        ssim = 0
        psnr = 0

        # Read the images
        shadow_img = cv.imread(input_shadow)
        no_shadow_img = cv.imread(input_no_shadow)

        # Binary Version of the two images
        shadow_img_bin = convert_to_binary(shadow_img)
        no_shadow_img_bin = convert_to_binary(no_shadow_img)

        cd = CD(shadow_img, no_shadow_img)
        sri_1, sri_2 = compute_SRI(shadow_img, no_shadow_img)
        mse = MSE(shadow_img_bin, no_shadow_img_bin)
        gmsd = GMSD(shadow_img, no_shadow_img)
        ssim, _ = ssim_fn(shadow_img_bin, no_shadow_img_bin, full=True)
        psnr = PSNR(shadow_img_bin, no_shadow_img_bin)

        # Print the results to the screen
        print(f"Single Image Evaluation: {input_shadow}")
        print("-" * 80)
        print(f"CD: {cd:.5f}")
        print(f"SRI: {(sri_2 - sri_1):.5f}")
        print(f"MSE: {mse:.5f}")
        print(f"GMSD: {gmsd:.5f}")
        print(f"SSIM: {ssim:.5f}")
        print(f"PSNR: {psnr:.5f}")
        print("-" * 80)

    # Batch of images
    elif mode == "directory":
        # TODO: Add code to evaluate a batch of images (directory)
        # All Metrics
        cd = []
        sri = []
        mse = []
        gmsd = []
        ssim = []
        psnr = []

        pass
    else:
        print(f"ERROR: {mode} is unrecognised.")


if __name__ == "__main__":
    main()
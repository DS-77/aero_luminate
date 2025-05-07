# AeroLuminate: Aerial Remote Sensing Image Shadow Detection and Removal

*Abstract*:

Remote sensing imagery is crucial for monitoring and planning, yet shadows frequently obscure details, particularly in urban environments. We present Aero-Luminate, a novel framework for shadow detection and removal tailored for aerial remote sensing data. Our approach leverages a two-stage pipeline: first, a transformer-based shadow detector adapted from the AdapterShadow architecture, followed by a diffusion model for shadow removal. This unified design aims to improve both accuracy and efficiency in automated processing. Preliminary results demonstrate Aero-Luminate's capabilities, achieving a Structural Similarity Index (SRI) of -0.01285 and Contrast Difference (CD) of 42.53835 compared to state-of-the-art methods like SDA, DC-ShadowNet, and DeS3-Deshadow. While further refinement is needed, these initial findings suggest the potential for Aero-Luminate to provide a valuable tool for enhancing remote sensing data analysis.

## Results

| Model                | SRI      | CD       | GMSD    | MSE     | SSIM    | PSNR     |
|----------------------|----------|----------|---------|---------|---------|----------|
| SDA                  | 0.00877  | 62.10790 | 0.26717 | 0.09692 | 0.66629 | 58.26689 |
| DC-ShadowNet         | -0.00058 | 35.34030 | 0.23306 | 0.08075 | 0.64809 | 59.05940 |
| DeS3-Deshadow        | 0.02063  | 50.01886 | 0.29791 | 0.28667 | 0.49903 | 53.55691 |
| Aero-Luminate (Ours) | -0.01285 | 42.53835 | 0.31720 | 0.46043 | 0.00754 | 51.52266 | 

## How to Install

It is recommended that you have Python 3.12 or greater.

#### Rocm

```commandline
conda create --name aero_shadow -f environment.yml
```

#### Nvidia
````commandline
conda create --name aero_shadow -f environment.yml
conda activate aero_shadow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
````

## How to Train

```commandline
python train_model.py -c <path to config file>

Example: python train_model.py -c configs/aero_luminate_config.yaml
```
If you would like to train our model on your own dataset, alter the file paths in the `configs/aero_luminate_config.yaml`.
You will also need to create a class for your dataset in the `utils/dataset.py`.

> You can find the weights and logs in a generated `runs` directory after executing the command.

> WARNING: It takes a long time to train.
## How to Run

```commandline
python test_model.py --adapter_weights weights/upload_weights/best_checkpoint_adapt_shadow.pth --diffusion_weights weights/upload_weights/best_checkpoint_shadow_diff.pth --diffusion_config configs/AISD_shadow_diff_config.json --input Data/AISD/test/shadow --output_dir runs/test_out --sam_checkpoint weights/adapter_weights/sam_vit_b_01ec64.pth --batch_size 1 --device cuda --time_benchmark
```

# Troubleshooting

All common issues will be posted here with their corresponding solutions. If you run into any errors, feel free to open an issue. We'll do our best to help you resolve it.

### Reference

We would like to thank the authors of [ShadowDiffusion](https://github.com/GuoLanqing/ShadowDiffusion) and [AdapterShadow](https://github.com/LeipingJie/AdapterShadow) for their work.
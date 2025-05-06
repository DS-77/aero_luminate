# AeroLuminate: Aerial Remote Sensing Image Shadow Detection and Removal

*Abstract*:



## Results

| Model                | SRI | CD | GMSD |
|----------------------|-----|----|------|
| SDA                  |     |    |      |
| DC-ShadowNet         |     |    |      |
| DeS3-Deshadow        |     |    |      |
| Aero-Luminate (Ours) |     |    |      |

## How to Install

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

## How to Run


# Troubleshooting

All common issues will be posted here with their corresponding solutions. If you run into any errors, feel free to open an issue. We'll do our best to help you resolve it.

### Reference

We would like to thank the authors of [ShadowDiffusion](https://github.com/GuoLanqing/ShadowDiffusion) and [AdapterShadow](https://github.com/LeipingJie/AdapterShadow) for their work.
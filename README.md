# DCENet

PyTorch inference code for **DCENet: Dual-Prior Conditioned Adaptive Expert Network for Low-Light Image Enhancement**.

This repository provides the inference implementation and pretrained checkpoint for low-light image enhancement.

## Overview

DCENet is designed for real-world low-light image enhancement. The model combines micro-structure prior mining and adaptive expert-based feature restoration to improve brightness, structural details, and perceptual quality under complex low-light degradation.

The current release includes:

- Inference code
- Network architecture
- Testing configuration
- Pretrained checkpoint for LOL-v1
- Environment configuration

Training code and extended evaluation scripts are not included in this release.

## Environment

Create the environment:

    conda env create -f environment.yml
    conda activate cwnet

If the Mamba-related packages cannot be installed automatically, local wheel files are provided:

    pip install causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install mamba_ssm-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

## Pretrained Model

The pretrained checkpoint is placed under:

    ./pretrained/DCENet_LOLv1.pth

## Data Preparation

Please organize the test data according to the paths defined in:

    ./options/test/LOL-v1.yml

You may modify the dataset paths in the YAML file according to your local directory.

## Testing

Run inference on LOL-v1:

    python test.py -opt ./options/test/LOL-v1.yml

The enhanced results will be saved to the output directory specified in the testing configuration.

## Repository Structure

    DCENet_release/
    ├── data/
    ├── models/
    ├── options/
    ├── pretrained/
    ├── utils/
    ├── test.py
    ├── environment.yml
    ├── README.md
    └── LICENSE

## Notes

This repository is released for academic research and reproducibility.

Some utility components are adapted from existing open-source frameworks. Please follow the corresponding license terms when using this code.

## Citation

The citation information will be updated after publication.

## Optional Mamba Wheel

The prebuilt `mamba_ssm` wheel is not included in this repository because of GitHub file-size limits.

If `mamba-ssm` cannot be installed from pip, please download the wheel from the release page or external link, and then install it manually:

    pip install mamba_ssm-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


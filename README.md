# Optimizing Camera Configurations for Multi-View Pedestrian Detection

## Overview
<!-- TODO -->

## Contents
  - [Dependencies](#dependencies)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)

## Dependencies
1. Please install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support with
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Then, install all other dependencies via
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
Running the **MVconfig** code requires the [CARLA simulator](https://carla.org/). We recommend using the CARLA Docker image.

1. To install the Docker container, please refer to the [Docker Engine installation guide](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. The CARLA Docker image is available on Docker Hub, which can be pulled using the following command
   ```bash
   docker pull carlasim/carla:0.9.14
   ```

## Usage
In order to train or test the detection and tracking model, as well as the camera control module, please follow the instructions below.

1. Train model with default configuration
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py -d carlax --reID --carla_gpu 0 --carla_cfg [cfg_name] --record_loss --carla_port 2000 --carla_tm_port 8000
   ```

2. Train model with three human expert configurations
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py -d carlax --reID --carla_gpu 0 --carla_cfg [cfg_name]_[1/2/3] --record_loss --carla_port 2000 --carla_tm_port 8000
   ```

3. Train model with interactive cameras and joint training
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py -d carlax --reID --interactive --carla_gpu 0 --carla_cfg [cfg_name] --epochs 50 --joint_training 1 --record_loss --carla_port 2000 --carla_tm_port 8000
   ```

4. Evaluate the trained model (with or without interactive cameras)
   ```bash
   CUDA_VISIBLE_DEVICES=0 python main.py -d carlax --reID [--interactive] --carla_gpu 0 --carla_cfg [cfg_name] --carla_port 2000 --carla_tm_port 8000 --eval --resume [log_path]
   ```

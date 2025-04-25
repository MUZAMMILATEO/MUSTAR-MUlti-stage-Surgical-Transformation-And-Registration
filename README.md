# MUSTAR: MUlti-stage Surgical Transformation And Registration

This repository provides the MUSTAR framework designed for surgical environments. It integrates Structure-from-Motion (SfM), pose estimation, and registration techniques to align pre-operative and intra-operative data.


# Getting Started
## Installation
```
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```
Check the system's CUDA version with nvcc
```
nvcc --version
```
Install pytorch with **matching** CUDA version following:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clone the repo and install the dependencies.
```
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

Setup the checkpoints for MASt3R and retrieval.  The license for the checkpoints and more information on the datasets used is written [here](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE).
```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## WSL Users
We have primarily tested on Ubuntu.  If you are using WSL, please checkout to the windows branch and follow the above installation.
```
git checkout windows
```
This disables multiprocessing which causes an issue with shared memory as discussed [here](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).

## 🛠️ Additional Installations for WSL Users

If you're running this project in a **WSL (Windows Subsystem for Linux)** environment, you need to install the following system dependencies to enable visualization and library support:

```bash
sudo apt install libglu1
sudo apt install libxcursor-dev
sudo apt install libxft2
sudo apt install libxinerama1
sudo apt install libfltk1.3-dev
sudo apt install libfreetype6-dev
sudo apt install libgl1-mesa-dev
sudo apt install libocct-foundation-dev
sudo apt install libocct-data-exchange-dev

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

## 📦 Additional Requirements for TEASER++ Registration

The MUSTAR framework uses TEASER++ for point cloud registration.  
If you have completed the above installation and created the conda environment, you **only need to install two additional packages** to enable TEASER++ functionalities:

```bash
pip install open3d==0.17.0
pip install teaserpp_python
```

📢 If you have not yet cloned the TEASER++ repository separately, you can refer to it [**here**](https://gitlab.nki.nl/igs-clinical-navigation/slam/teaser-plusplus) for pre-intra-teaserpp branch compilation and advanced usage.



Run the following in your WSL terminal:

## Data Preparation

### 1. Pre-operative Point Cloud

Place the following pre-operative point clouds in the `./SfM-SLAM/pre_cloud/` directory:

- `Kyoto_CT.ply` - Pre-operative point cloud focused on the region of interest for registration.
- `Kyoto_CT_with_interior.ply` - Full pre-operative point cloud including the interior structures (such as vessels, lesions, etc.).

### 2. Structure-from-Motion (SfM) Point Cloud

Place the SfM-generated point cloud in the `./SfM-SLAM/sfm_pts/` directory as:

- `points3D.ply` - The reconstructed point cloud generated from SfM such as InstantSplat.

---

## 3. Generate Image Sequence Metadata

To generate a .txt file corresponding to your input image sequence:
```
python ./extra_tools/txt_gen.py
```

When prompted, provide the absolute path to your folder containing RGB images.

## 4. Folder Structure Overview
```
MASt3R-SLAM/
├── SfM-SLAM/
│   ├── pre_cloud/
│   │   ├── Kyoto_CT.ply
│   │   └── Kyoto_CT_with_interior.ply
│   └── sfm_pts/
│       ├── points3D.ply
│       └── keyframe0.ply (after running main.py)
├── config/
│   └── base.yaml
├── datasets/
│   └── tum/
│       └── rgbd_dataset/
└── extra_tools/
    └── txt_gen.py
```

## 🚀 Running the SLAM Framework

Run the main pipeline with the following command:

```bash
python main.py --dataset /path/to/your/rgbd_dataset/ --config config/base.yaml
```

**Note:** After successful execution, the first keyframe will be saved as keyframe0.ply in the ./SfM-SLAM/sfm_pts/ folder.



## Live Demo
Connect a realsense camera to the PC and run
```
python main.py --dataset realsense --config config/base.yaml
```
## Running on a video
Our system can process either MP4 videos or folders containing RGB images.
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml
```
If the calibration parameters are known, you can specify them in intrinsics.yaml
```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml --calib config/intrinsics.yaml
python main.py --dataset <path/to/folder> --config config/base.yaml --calib config/intrinsics.yaml
```

## Downloading Dataset
### TUM-RGBD Dataset
```
bash ./scripts/download_tum.sh
```

### 7-Scenes Dataset
```
bash ./scripts/download_7_scenes.sh
```

### EuRoC Dataset
```
bash ./scripts/download_euroc.sh
```
### ETH3D SLAM Dataset
```
bash ./scripts/download_eth3d.sh
```

## Running Evaluations
All evaluation script will run our system in a single-threaded, headless mode.
We can run evaluations with/without calibration:
### TUM-RGBD Dataset
```
bash ./scripts/eval_tum.sh 
bash ./scripts/eval_tum.sh --no-calib
```

### 7-Scenes Dataset
```
bash ./scripts/eval_7_scenes.sh 
bash ./scripts/eval_7_scenes.sh --no-calib
```

### EuRoC Dataset
```
bash ./scripts/eval_euroc.sh 
bash ./scripts/eval_euroc.sh --no-calib
```
### ETH3D SLAM Dataset
```
bash ./scripts/eval_eth3d.sh 
```

## Reproducibility
There might be minor differences between the released version and the results in the paper after developing this multi-processing version. 
We run all our experiments on an RTX 4090, and the performance may differ when running with a different GPU.

## Acknowledgement
We sincerely thank the developers and contributors of the many open-source projects that our code is built upon.
- [MASt3R](https://github.com/naver/mast3r)
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
      To be added
```

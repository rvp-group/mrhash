<p align="center">
  <h2 align="center">Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids</h2>
  <p align="center">
    <strong>Lorenzo De Rebotti</strong>
    ·
    <strong>Emanuele Giacomini</strong>
    ·
    <strong>Giorgio Grisetti</strong>
    ·
    <strong>Luca Di Giammarino</strong>
  </p>
</p>
<h3 align="center">ACM Transactions on Graphics</h3>

<p align="center"><b>MrHash</b> is a GPU-accelerated 3D reconstruction pipeline that uses variance-adaptive voxel hashing for efficient TSDF fusion with optional 3D Gaussian Splatting rendering.</p>

<p align="center">
  <img src="./room0_readme.gif"</>
</p>



# Installation

## Prerequisites

Before you begin, ensure you have an NVIDIA GPU with CUDA capabilities 

### Clone the Repository

First, clone this repository along with all its submodules:

```sh
git clone --recursive https://github.com/rvp-group/mrhash.git
cd mrhash
```

## Installation via Pixi (Recommended)

If you have [Pixi](https://pixi.sh) installed, setting up the environment is straightforward:

```sh
pixi install
```

Once the environment is set up, activate the Pixi shell to proceed with the next steps:

```sh
pixi shell
```

## Manual Installation

If you prefer to build from scratch or don't use Pixi, follow these steps:

### Prerequisites
- CUDA 12.6 or later installed on your system
- Python 3.8+ with pip

### Step 1: Install LibTorch

Download and extract LibTorch with CUDA support:

```sh
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.8.0+cu126.zip -d third_party
```

### Step 2: Install the Python Package

Install the library in development mode:

```sh
pip install -e .
```

# Usage

This project includes pre-configured settings and scripts to get you started quickly with various datasets.

## Supported Datasets and Formats

We provide ready-to-use configurations in the `configurations/` folder for:
- **RGB-D datasets**: Replica, ScanNet
- **LiDAR datasets**: VBR, Newer College Dataset, Oxford Spires

Supported data formats include:
- ROS bags (single or multiple in a folder)
- RGB-D image sequences (with depth in `.png` format and rgb in `.jpg` in the same folder called `results`)
- Point clouds (`.ply` format)
- KITTI format

## Running Examples

Navigate to the `apps/` folder and choose the appropriate script for your data format:

### RGB-D Reconstruction

For standard TSDF-based reconstruction (e.g., Replica dataset):

```sh
python rgbd_runner.py
```

### RGB-D with Gaussian Splatting

For 3D reconstruction combined with 3D Gaussian Splatting rendering:

```sh
python3 rgbd_gs_runner.py
```
>[!NOTE]
> Gaussian Splatting requires significantly more GPU memory. Ensure your system has sufficient VRAM available.

## Python API

The pipeline is exposed via Python bindings:

```python
import yaml
import numpy as np
from mrhash.src.pygeowrapper import GeoWrapper

# Load configuration from YAML file
with open("configurations/replica.cfg", "r") as f:
    config = yaml.safe_load(f)

# Create GeoWrapper instance with parameters
geo_wrapper = GeoWrapper(
    sdf_truncation=config["map"]["sdf_truncation"],
    sdf_truncation_scale=config["map"]["sdf_truncation_scale"],
    integration_weight_sample=config["map"]["integration_weight_sample"],
    virtual_voxel_size=config["map"]["virtual_voxel_size"],
    n_frames_invalidate_voxels=config["map"]["n_frames_invalidate_voxels"],
    voxel_extents_scale=config["streamer"]["voxel_extents_scale"],
    viewer_active=False,
    marching_cubes_threshold=config["mesh"]["marching_cubes_threshold"],
    min_weight_threshold=config["mesh"]["min_weight_threshold"],
    min_depth=config["sensor"]["min_depth"],
    max_depth=config["sensor"]["max_depth"],
)

# Set camera intrinsics
geo_wrapper.setCamera(fx, fy, cx, cy, img_rows, img_cols, min_depth, max_depth, camera_model)

# Process frames in a loop
for depth_image, rgb_image, pose, quaternion in your_data_loader:
    # Set current camera pose
    geo_wrapper.setCurrPose(pose, quaternion)
    
    # Set input data (depth and RGB images as numpy arrays)
    geo_wrapper.setDepthImage(depth_image)  # numpy array (H, W) with float32
    geo_wrapper.setRGBImage(rgb_image)      # numpy array (H, W, 3) with uint8
    
    # Integrate current frame into the voxel grid
    geo_wrapper.compute()

# Extract final mesh after processing all frames
geo_wrapper.extractMesh("output_mesh.ply")

# Optional: Save hash table and voxel data for visualization
geo_wrapper.serializeData("hash_points.ply", "voxel_points.ply")
```

### Using Gaussian Splatting

To enable 3D Gaussian Splatting rendering, simply pass the `gs_optimization_param_path` parameter when creating the `GeoWrapper`:

```python
# Enable Gaussian Splatting by providing the optimization parameters
geo_wrapper = GeoWrapper(
    sdf_truncation=config["map"]["sdf_truncation"],
    # ... other parameters ...
    gs_optimization_param_path="apps/params.json",  # Enable GS with this parameter
    min_depth=config["sensor"]["min_depth"],
    max_depth=config["sensor"]["max_depth"],
)

# Process frames as usual
for depth_image, rgb_image, pose, quaternion in your_data_loader:
    geo_wrapper.setCurrPose(pose, quaternion)
    geo_wrapper.setDepthImage(depth_image)
    geo_wrapper.setRGBImage(rgb_image)
    geo_wrapper.compute()

# Save the Gaussian Splatting point cloud
geo_wrapper.GSSavePointCloud("output_gs_model")
```

For a complete working example, see `apps/rgbd_gs_runner.py`.

### Key Functions

- **`GeoWrapper(...)`**: Initialize the reconstruction system with configuration parameters
- **`setCamera(...)`**: Configure camera intrinsics and distortion model
- **`setCurrPose(...)`**: Set the current camera pose (translation + quaternion)
- **`setDepthImage(...)`** / **`setRGBImage(...)`**: Provide input depth and RGB data
- **`compute()`**: Integrate the current frame into the TSDF volume
- **`extractMesh(...)`**: Generate and save a triangle mesh using marching cubes
- **`GSSavePointCloud(...)`**: Export the 3D Gaussian Splatting model (requires `gs_optimization_param_path` set)
- **`serializeData(...)`**: Export voxel grid data for debugging or visualization
- **`serializeGrid(...)`**: Save the entire hash table and voxel grid to disk for later resumption
- **`deserializeGrid(...)`**: Load a previously saved grid state from disk

## Evaluation

To evaluate the quality of the reconstructed mesh against a ground truth point cloud, use the `apps/eval_reconstruction.py` script.

### Basic Usage

```sh
python apps/eval_reconstruction.py evaluate <reference_pcd> <target_mesh>... [OPTIONS]
```

- `reference_pcd`: Path to the ground truth point cloud (e.g., `.ply` file).
- `target_mesh`: One or more paths to the reconstructed meshes to evaluate.

### Options

- `--crop`: Perform cropping of the reference point cloud to the union of target meshes. Requires `--out-ref-crop`.
- `--out-ref-crop`, `-o`: Path to save the cropped reference point cloud. Mandatory if `--crop` is used.
- `--visualize`, `-v`: Visualize the alignment and error map.
- `--thresholds`: List of distance thresholds for precision/recall (default: `[0.05, 0.1, 0.2, 0.25, 0.5]`).
- `--truncation-acc-thresholds`: List of truncation thresholds for accuracy (default: `[0.10, 0.2, 0.4, 0.5, 1.0]`).
- `--cropping-distance`: Distance threshold for cropping (default: `1.0`).

### Example

**Standard Evaluation:**
```sh
python apps/eval_reconstruction.py evaluate data/gt.ply data/mesh.ply
```

**Evaluation with Cropping:**
```sh
python apps/eval_reconstruction.py evaluate data/gt.ply data/mesh.ply --crop -o data/gt_cropped.ply
```

**Evaluation with Visualization:**
```sh
python apps/eval_reconstruction.py evaluate data/gt.ply data/mesh.ply -v
```

This will generate an `evaluation_metrics.csv` file in the same directory as the reference file used (either `reference_pcd` or `out_ref_crop`).


# Citation

If you find this work useful in your research, please consider citing our paper:

```
@article{10.1145/3777909,
author = {De Rebotti, Lorenzo and Giacomini, Emanuele and Grisetti, Giorgio and Di Giammarino, Luca},
title = {Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {0730-0301},
url = {https://doi.org/10.1145/3777909},
doi = {10.1145/3777909},
journal = {ACM Trans. Graph.},
keywords = {Surface Reconstruction, Novel View Synthesis, Gaussian Splatting}}
```

# Acknowledgments

We gratefully acknowledge the contributions of the following open-source projects, which have been instrumental in the development of this work:

- [VoxelHashing](https://github.com/niessner/VoxelHashing): The pipeline logic is inspired by the core hashing algorithms.
- [LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio): For their efficient implementation of 3D Gaussian Splatting.

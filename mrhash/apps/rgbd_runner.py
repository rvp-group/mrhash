import shutil
import sys
import time
from pathlib import Path

import numpy as np
import typer
import yaml
from mrhash.src.pygeowrapper import GeoWrapper
from rich.console import Console
from tqdm import tqdm
from typing_extensions import Annotated

from utils.camera import Camera, CameraModel
from utils.depth_reader import DepthReader

console = Console()


def main(
    config_path: Annotated[
        str, typer.Argument(help="Path of the config file")
    ] = "../configurations/replica.cfg",
) -> None:

    config = Path(config_path)

    if not config.exists():
        console.print(f"[red]Error: Config file {config} does not exist!")
        sys.exit(1)
    with open(config, "r") as data_config_file:
        data_cf = yaml.safe_load(data_config_file)
    data_path = Path(data_cf["data_path"])
    results_dir = Path(data_cf["results_path"])
    if not data_path.exists():
        console.print(f"[red]Error: Data path {data_path} does not exist!")
        sys.exit(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    copied_config = results_dir / f"{timestamp}_{config.name}"
    shutil.copy(config, copied_config)

    sdf_truncation = data_cf["map"]["sdf_truncation"]
    sdf_truncation_scale = data_cf["map"]["sdf_truncation_scale"]
    integration_weight_sample = data_cf["map"]["integration_weight_sample"]
    virtual_voxel_size = data_cf["map"]["virtual_voxel_size"]
    n_frames_invalidate_voxels = data_cf["map"]["n_frames_invalidate_voxels"]

    voxel_extents_scale = data_cf["streamer"]["voxel_extents_scale"]

    marching_cubes_threshold = data_cf["mesh"]["marching_cubes_threshold"]
    min_weight_threshold = data_cf["mesh"]["min_weight_threshold"]
    sdf_var_threshold = data_cf["mesh"]["sdf_var_threshold"]
    vertices_merging_threshold = data_cf["mesh"]["vertices_merging_threshold"]

    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = data_cf["sensor"]["intrinsics"][0]
    K[1, 1] = data_cf["sensor"]["intrinsics"][1]
    K[0, 2] = data_cf["sensor"]["intrinsics"][2]
    K[1, 2] = data_cf["sensor"]["intrinsics"][3]
    K[2, 2] = 1

    img_rows = data_cf["sensor"]["resolution"][1]
    img_cols = data_cf["sensor"]["resolution"][0]
    min_depth = data_cf["sensor"]["min_depth"]
    max_depth = data_cf["sensor"]["max_depth"]
    depth_scaling = data_cf["sensor"]["depth_scaling"]
    sensor_hz = data_cf["sensor"]["hz"]

    reader = DepthReader(
        data_path,
        min_range=min_depth,
        max_range=max_depth,
        depth_scaling=depth_scaling,
        sensor_hz=sensor_hz,
    )

    end_frame = data_cf["end_frame"] if data_cf["end_frame"] != -1 else len(reader) + 1

    console.print(f"[yellow] sdf_truncation: {sdf_truncation}")
    console.print(f"[yellow] sdf_truncation_scale: {sdf_truncation_scale}")
    console.print(f"[yellow] integration_weight_sample: {integration_weight_sample}")
    console.print(f"[yellow] virtual_voxel_size: {virtual_voxel_size}")
    console.print(f"[yellow] n_frames_invalidate_voxels: {n_frames_invalidate_voxels}")
    console.print(f"[yellow] voxel_extents_scale: {voxel_extents_scale}")
    console.print(f"[yellow] min_weight_threshold: {min_weight_threshold}")
    console.print(f"[yellow] sdf_var_threshold: {sdf_var_threshold}")
    console.print(f"[yellow] marching_cubes_threshold: {marching_cubes_threshold}")

    console.print(f"[yellow] min depth: {min_depth} [m]")
    console.print(f"[yellow] max depth: {max_depth} [m]")
    console.print(f"[yellow] img_rows: {img_rows} [m]")
    console.print(f"[yellow] img_cols: {img_cols} [m]")
    console.print(f"[yellow] depth_scaling: {depth_scaling} [m]")
    console.print(f"[yellow] num_scan: {len(reader)}")

    rgbd_camera = Camera(
        rows=img_rows,
        cols=img_cols,
        K=K,
        min_depth=min_depth,
        max_depth=max_depth,
        model=CameraModel.Pinhole,
    )

    geo_wrapper = GeoWrapper(
        sdf_truncation=sdf_truncation,
        sdf_truncation_scale=sdf_truncation_scale,
        integration_weight_sample=integration_weight_sample,
        virtual_voxel_size=virtual_voxel_size,
        n_frames_invalidate_voxels=n_frames_invalidate_voxels,
        voxel_extents_scale=voxel_extents_scale,
        viewer_active=False,
        marching_cubes_threshold=marching_cubes_threshold,
        min_weight_threshold=min_weight_threshold,
        sdf_var_threshold=sdf_var_threshold,
        vertices_merging_threshold=vertices_merging_threshold,
        projective_sdf=True,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    geo_wrapper.setCamera(
        rgbd_camera.fx_,
        rgbd_camera.fy_,
        rgbd_camera.cx_,
        rgbd_camera.cy_,
        rgbd_camera.rows_,
        rgbd_camera.cols_,
        rgbd_camera.min_depth_,
        rgbd_camera.max_depth_,
        rgbd_camera.model_,
    )

    for frame, pose, quat, depth_img, rgb_img in tqdm(reader, desc="processing..."):
        if frame > end_frame:
            break
        geo_wrapper.setCurrPose(pose, quat)
        geo_wrapper.setDepthImage(depth_img)
        geo_wrapper.setRGBImage(rgb_img)
        geo_wrapper.compute()

    geo_wrapper.streamAllOut()
    geo_wrapper.extractMesh(f"{results_dir}/mesh_{timestamp}.ply")
    geo_wrapper.serializeData(
        f"{results_dir}/hash_points_{timestamp}.ply",
        f"{results_dir}/voxel_points_{timestamp}.ply",
    )
    geo_wrapper.clearBuffers()


def run():
    typer.run(main)


if __name__ == "__main__":
    run()

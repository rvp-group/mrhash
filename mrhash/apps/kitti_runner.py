import os
import shutil
import sys
import time
from pathlib import Path

import typer
import yaml
from rich.console import Console
from tqdm import tqdm
from typing_extensions import Annotated

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from mrhash.src.pygeowrapper import GeoWrapper

from utils.kitti_reader import KittiReader

console = Console()


def main(
    config_path: Annotated[
        str,
        typer.Argument(help="Path of the config file"),
    ] = "../configurations/maicity.cfg",
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
    min_weight_threshold = data_cf["mesh"]["min_weight_threshold"]
    sdf_var_threshold = data_cf["mesh"]["sdf_var_threshold"]
    vertices_merging_threshold = data_cf["mesh"]["vertices_merging_threshold"]
    marching_cubes_threshold = data_cf["mesh"]["marching_cubes_threshold"]

    min_depth = data_cf["sensor"]["min_depth"]
    max_depth = data_cf["sensor"]["max_depth"]
    sensor_hz = data_cf["sensor"]["hz"]

    reader = KittiReader(
        data_path, min_range=min_depth, max_range=max_depth, sensor_hz=sensor_hz
    )

    end_frame = data_cf["end_frame"] if data_cf["end_frame"] != -1 else len(reader) + 1

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
    console.print(f"[yellow] num_scan: {len(reader)}")

    frame_idx = 0
    for t, quat, points in tqdm(reader, desc="processing"):
        if frame_idx > end_frame:
            break
        geo_wrapper.setCurrPose(t, quat)
        geo_wrapper.setPointCloud(points[:, :3], False)
        geo_wrapper.compute()
        frame_idx += 1

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

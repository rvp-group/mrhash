from pathlib import Path
from typing import List, Optional
import csv
import open3d as o3d

import typer

from utils.eval_utils import crop_union, eval_mesh, eval_mesh_thresholds

app = typer.Typer()


@app.command()
def crop(ref_pcd: Path, target_mesh: List[Path], out_ref_crop: Path):
    crop_union(ref_pcd, target_mesh, out_ref_crop)


@app.command("evaluate")
def evaluate(
    ref_pcd: Path,
    target_meshes: List[Path],
    out_ref_crop: Optional[Path] = typer.Option(
        None,
        "--out-ref-crop",
        "-o",
        help="Path to save cropped reference point cloud. Required if --crop is set.",
    ),
    thresholds: List[float] = typer.Option(
        [0.05, 0.1, 0.2, 0.25, 0.5], help="Distance thresholds for precision/recall"
    ),
    truncation_acc_thresholds: List[float] = typer.Option(
        [0.10, 0.2, 0.4, 0.5, 1.0], help="Truncation thresholds for accuracy"
    ),
    cropping_distance: float = typer.Option(
        1.00, help="Distance threshold for cropping"
    ),
    perform_crop: bool = typer.Option(
        False, "--crop", help="Whether to perform cropping before evaluation"
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Visualize the alignment and error map"
    ),
):
    reference_to_use = ref_pcd

    if perform_crop:
        if out_ref_crop is None:
            typer.echo(
                "Error: --out-ref-crop must be provided when --crop is used.", err=True
            )
            raise typer.Exit(code=1)
        crop_union(ref_pcd, target_meshes, out_ref_crop, dist_thre=cropping_distance)
        reference_to_use = out_ref_crop

    all_metrics = []

    for mesh in target_meshes:
        metrics_dict, error_map = eval_mesh_thresholds(
            mesh,
            reference_to_use,
            generate_error_map=True,
            down_sample_res=0,
            threshold_list=thresholds,
            truncation_acc_list=truncation_acc_thresholds,
            truncation_com=cropping_distance,
            visualize=visualize,
        )
        if visualize:
            print(f"Visualizing error map for {mesh.name}...")
            o3d.visualization.draw_geometries(
                [error_map], window_name=f"Error Map: {mesh.name}"
            )
        for (threshold, trunc_acc), metrics in metrics_dict.items():
            row = {
                "mesh": mesh.stem,
                "threshold": threshold,
                "truncation_acc": trunc_acc,
                **metrics,
            }
            all_metrics.append(row)

    csv_path = reference_to_use.parent / "evaluation_metrics.csv"
    if all_metrics:
        # Define fixed header order
        fieldnames = ["mesh", "threshold", "truncation_acc"]
        # Add remaining keys from the first result
        remaining_keys = [k for k in all_metrics[0].keys() if k not in fieldnames]
        fieldnames.extend(sorted(remaining_keys))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)


if __name__ == "__main__":
    app()

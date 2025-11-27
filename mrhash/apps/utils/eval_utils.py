import numpy as np
from rich.progress import track
import open3d as o3d
from typing import List
from pathlib import Path


def eval_mesh_thresholds(
    file_pred: Path,
    file_trgt: Path,
    down_sample_res: float = 0.02,
    threshold_list: List[float] = [0.2],
    truncation_acc_list: List[float] = [0.50],
    truncation_com: float = 0.50,
    gt_bbx_mask_on: bool = True,
    mesh_sample_point: int = 10_000_000,
    align_mesh: bool = False,
    generate_error_map: bool = False,
    visualize: bool = False,
):
    """Compute Mesh metrics between prediction and target given a list of thesholds
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        thresholds: a list of distance thresholds used to compute precision/recall
        truncation_acc: list of thresholds for which points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: list of thresholds for which points points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """
    print(f"Reading mesh from {file_pred}")
    mesh_pred = o3d.io.read_triangle_mesh(str(file_pred))
    print(f"Reading point cloud from {file_trgt}")
    pcd_trgt = o3d.io.read_point_cloud(str(file_trgt))

    if gt_bbx_mask_on:
        print("Filtering prediction outside the target bounding box")
        trgt_bbx = pcd_trgt.get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound()
        min_bound[2] -= down_sample_res
        max_bound = trgt_bbx.get_max_bound()
        max_bound[2] += down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        mesh_pred = mesh_pred.crop(trgt_bbx)

    chunk_size = 1_000_000
    num_chunks = int(np.ceil(mesh_sample_point / chunk_size))
    points_list = []

    for i in track(range(num_chunks), description="Uniformly sampling mesh"):
        n_points = min(chunk_size, mesh_sample_point - i * chunk_size)
        pcd_chunk = mesh_pred.sample_points_uniformly(number_of_points=n_points)
        points_list.append(np.asarray(pcd_chunk.points))

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(np.concatenate(points_list))

    if down_sample_res > 0:
        print(f"Down sampling mesh to {down_sample_res} m ")
        pred_pt_count_before = len(pcd_pred.points)
        pcd_pred = pcd_pred.voxel_down_sample(down_sample_res)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample_res)
        pred_pt_count_after = len(pcd_pred.points)
        print(
            f"Predicted mesh uniform sample: {pred_pt_count_before} -> {pred_pt_count_after} ({down_sample_res})"
        )

    pcd_pred.paint_uniform_color([1, 0.706, 0])
    pcd_trgt.paint_uniform_color([0, 0.651, 0.929])
    if visualize:
        o3d.visualization.draw_geometries(
            [pcd_pred, pcd_trgt], window_name="Alignment Check"
        )

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    _, dist_r = nn_correspondence(
        verts_pred,
        verts_trgt,
        truncation_com,
        False,
        verbose=True,
        description="Computing correspondences <prediction, target>",
    )  # find nn in predict samples for each ground truth sample -> recall related
    metrics = {}
    error_maps = {}
    for threshold, truncation_acc in zip(threshold_list, truncation_acc_list):
        _, dist_p = nn_correspondence(
            verts_trgt,
            verts_pred,
            truncation_acc,
            True,
            verbose=True,
            description="Computing correspondences <target, prediction>",
        )  # find nn in ground truth samples for each predict sample -> precision related
        dist_p = np.array(dist_p)
        dist_r = np.array(dist_r)

        error_map = o3d.geometry.PointCloud()
        if generate_error_map:
            error_map = generate_save_error_map(verts_trgt, dist_r)

        dist_p_s = np.square(dist_p)
        dist_r_s = np.square(dist_r)

        dist_p_mean = np.mean(dist_p)
        dist_r_mean = np.mean(dist_r)

        dist_p_s_mean = np.mean(dist_p_s)
        dist_r_s_mean = np.mean(dist_r_s)

        chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
        chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

        precision = np.mean((dist_p < threshold).astype("float")) * 100.0  # %
        recall = np.mean((dist_r < threshold).astype("float")) * 100.0  # %
        fscore = 2 * precision * recall / (precision + recall)  # %

        metrics[threshold, truncation_acc] = {
            "MAE_accuracy (cm)": float(dist_p_mean * 100),
            "MAE_completeness (cm)": float(dist_r_mean * 100),
            "Chamfer_L1 (cm)": float(chamfer_l1 * 100),
            "Precision [Accuracy] (%)": float(precision),
            "Recall [Completeness] (%)": float(recall),
            "F-score (%)": float(fscore),
            "Inlier_threshold (m)": float(threshold),  # evlaution setup
            "Outlier_truncation_acc (m)": float(truncation_acc),  # evlaution setup
            "Outlier_truncation_com (m)": float(truncation_com),  # evlaution setup
        }
        print(metrics[threshold, truncation_acc])
        error_maps[threshold, truncation_acc] = error_map
    return metrics, error_map


# Mapping evaluation is borrowed from N3-Mapping (https://github.com/tiev-tongji/N3-Mapping/blob/main/eval/eval_utils.py)
def eval_mesh(
    file_pred: Path,
    file_trgt: Path,
    down_sample_res: float = 0.02,
    threshold: float = 0.2,
    truncation_acc: float = 0.50,
    truncation_com: float = 0.50,
    gt_bbx_mask_on: bool = True,
    mesh_sample_point: int = 10_000_000,
    align_mesh: bool = False,
    generate_error_map: bool = False,
    visualize: bool = False,
):
    """Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """
    print(f"Reading mesh from {file_pred}")
    mesh_pred = o3d.io.read_triangle_mesh(str(file_pred))
    print(f"Reading point cloud from {file_trgt}")
    pcd_trgt = o3d.io.read_point_cloud(str(file_trgt))

    if gt_bbx_mask_on:
        print("Filtering prediction outside the target bounding box")
        trgt_bbx = pcd_trgt.get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound()
        min_bound[2] -= down_sample_res
        max_bound = trgt_bbx.get_max_bound()
        max_bound[2] += down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        mesh_pred = mesh_pred.crop(trgt_bbx)

    print("Uniformly sampling mesh")
    chunk_size = 1_000_000
    num_chunks = int(np.ceil(mesh_sample_point / chunk_size))
    points_list = []

    for i in track(range(num_chunks), description="Uniformly sampling mesh"):
        n_points = min(chunk_size, mesh_sample_point - i * chunk_size)
        pcd_chunk = mesh_pred.sample_points_uniformly(number_of_points=n_points)
        points_list.append(np.asarray(pcd_chunk.points))

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(np.concatenate(points_list))

    if down_sample_res > 0:
        print(f"Down sampling mesh to {down_sample_res} m ")
        pred_pt_count_before = len(pcd_pred.points)
        pcd_pred = pcd_pred.voxel_down_sample(down_sample_res)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample_res)
        pred_pt_count_after = len(pcd_pred.points)
        print(
            f"Predicted mesh uniform sample: {pred_pt_count_before} -> {pred_pt_count_after} ({down_sample_res})"
        )

    pcd_pred.paint_uniform_color([1, 0.706, 0])
    pcd_trgt.paint_uniform_color([0, 0.651, 0.929])
    if visualize:
        o3d.visualization.draw_geometries(
            [pcd_pred, pcd_trgt], window_name="Alignment Check"
        )

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    _, dist_p = nn_correspondence(
        verts_trgt,
        verts_pred,
        truncation_acc,
        True,
        verbose=True,
        description="Computing correspondences <target, prediction>",
    )  # find nn in ground truth samples for each predict sample -> precision related
    _, dist_r = nn_correspondence(
        verts_pred,
        verts_trgt,
        truncation_com,
        False,
        verbose=True,
        description="Computing correspondences <prediction, target>",
    )  # find nn in predict samples for each ground truth sample -> recall related
    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)

    error_map = o3d.geometry.PointCloud()
    if generate_error_map:
        error_map = generate_save_error_map(verts_trgt, dist_r)

    dist_p_s = np.square(dist_p)
    dist_r_s = np.square(dist_r)

    dist_p_mean = np.mean(dist_p)
    dist_r_mean = np.mean(dist_r)

    dist_p_s_mean = np.mean(dist_p_s)
    dist_r_s_mean = np.mean(dist_r_s)

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

    precision = np.mean((dist_p < threshold).astype("float")) * 100.0  # %
    recall = np.mean((dist_r < threshold).astype("float")) * 100.0  # %
    fscore = 2 * precision * recall / (precision + recall)  # %

    metrics = {
        "MAE_accuracy (cm)": dist_p_mean * 100,
        "MAE_completeness (cm)": dist_r_mean * 100,
        "Chamfer_L1 (cm)": chamfer_l1 * 100,
        "Precision [Accuracy] (%)": precision,
        "Recall [Completeness] (%)": recall,
        "F-score (%)": fscore,
        "Inlier_threshold (m)": threshold,  # evlaution setup
        "Outlier_truncation_acc (m)": truncation_acc,  # evlaution setup
        "Outlier_truncation_com (m)": truncation_com,  # evlaution setup
    }
    return metrics, error_map


def generate_save_error_map(points, errors):
    errors = np.clip(errors, 0, 0.20) / 0.2
    # errors = np.clip(errors, 0, 0.05) / 0.05
    colors = colormap(errors)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def generate_mesh_error_map(file_pred, file_trgt, tr=0.50):
    mesh_pred = o3d.io.read_triangle_mesh(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)

    mesh_verts_pred = np.asarray(mesh_pred.vertices)
    verts_trgt = np.asarray(pcd_trgt.points)
    _, acc_dist = nn_correspondence(
        verts_trgt, mesh_verts_pred, tr, False
    )  # find nn in ground truth samples for each predict sample -> precision related
    normal_errors = np.clip(acc_dist, 0, 0.1) / 0.1  # set error interval
    colors = colormap(normal_errors)
    mesh_pred.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh_pred


def colormap(errors):
    colors = np.zeros((len(errors), 3))
    colors[:, 0] = 1.0
    colors[:, 1] = 1 - errors
    colors[:, 2] = 1 - errors

    return colors


def nn_correspondence(
    verts1,
    verts2,
    truncation_dist,
    ignore_outlier=True,
    verbose=False,
    description="Computing correspondences",
):
    """for each vertex in verts2 find the nearest vertex in verts1
    Args:
        verts1: nx3 np.array (target/reference)
        verts2: mx3 np.array (source/query)
        truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
    Returns:
        (None, [distances])
    """
    if len(verts1) == 0 or len(verts2) == 0:
        return None, []

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(verts1)

    chunk_size = 1_000_000
    dists_list = []

    indices = range(0, len(verts2), chunk_size)
    if verbose:
        indices = track(indices, description=description)

    for i in indices:
        chunk = verts2[i : i + chunk_size]
        pcd_chunk = o3d.geometry.PointCloud()
        pcd_chunk.points = o3d.utility.Vector3dVector(chunk)
        dists_chunk = np.asarray(pcd_chunk.compute_point_cloud_distance(pcd1))
        dists_list.append(dists_chunk)

    dists = np.concatenate(dists_list)

    if ignore_outlier:
        mask = dists < truncation_dist
        return None, dists[mask]
    else:
        dists[dists > truncation_dist] = truncation_dist
        return None, dists


def crop_union(
    file_gt: Path,
    files_pred: List[Path],
    out_file_crop: Path,
    dist_thre=1.2,
    mesh_sample_point=10_000_000,
):
    """Get the union of ground truth point cloud according to the intersection of the predicted
    mesh by different methods
    Args:
        file_gt: file path of the ground truth (should be point cloud)
        files_pred: a list of the paths of different methods' reconstruction (should be mesh)
        out_file_crop: output path of the cropped ground truth point cloud
        dist_thre: nearest neighbor distance threshold in meter
        mesh_sample_point: number of the sampling points from the mesh
    """
    print("Load the original ground truth point cloud from:", file_gt)
    pcd_gt = o3d.io.read_point_cloud(file_gt)
    pcd_gt_pts = np.asarray(pcd_gt.points)

    sampled_pcds = [
        o3d.io.read_triangle_mesh(file_pred).sample_points_uniformly(mesh_sample_point)
        for file_pred in track(files_pred, "Processing Meshes...")
    ]

    merged_pcd_pts = np.vstack([np.asarray(pcd.points) for pcd in sampled_pcds])

    print("Building KDTree with sampled points...")
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(merged_pcd_pts)

    print("Finding nearest neighbors for ground truth points...")
    dists = np.asarray(pcd_gt.compute_point_cloud_distance(sampled_pcd))
    near_mask = dists < dist_thre

    union_pcd_gt_pts = pcd_gt_pts[near_mask]

    crop_pcd_gt = o3d.geometry.PointCloud()
    crop_pcd_gt.points = o3d.utility.Vector3dVector(union_pcd_gt_pts)

    print("Output the cropped ground truth to:", out_file_crop)
    o3d.io.write_point_cloud(out_file_crop, crop_pcd_gt)

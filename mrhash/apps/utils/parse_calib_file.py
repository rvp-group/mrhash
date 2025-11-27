import yaml
import cv2
import numpy as np
from typing import Tuple


def read_extrinsics(f: str):
    """Returns Camera in LiDAR with Rodrigues representation

    Args:
        f (str): _description_

    Returns:
        (rvec_cTl, tvec_cTl) : Relative offset of LiDAR with respect to Camera frame
        (rvec_lTc, tvec_lTc) : Relative offset of Camera with respect to LiDAR frame
    """
    # Read yaml
    ydict = {}
    with open(f, "r") as fin:
        ydict = yaml.safe_load(fin)

    lidar_T_camera = np.float32(ydict["cam_r"]["T_b"])

    # Invert camera_T_lidar
    # lidar_T_camera = np.linalg.inv(camera_T_lidar)
    rvec_lTc, _ = cv2.Rodrigues(lidar_T_camera[:3, :3])
    rvec_lTc = np.float32(rvec_lTc).flatten()

    camera_T_lidar = np.linalg.inv(lidar_T_camera)
    rvec_cTl, _ = cv2.Rodrigues(camera_T_lidar[:3, :3])
    rvec_cTl = np.float32(rvec_cTl).flatten()

    # Extract rvec and tvec
    return rvec_cTl, camera_T_lidar[:3, 3], rvec_lTc, lidar_T_camera[:3, 3]


def read_intrinsics(f: str) -> Tuple[np.float32]:
    """Returns camera intrinsics

    Args:
        f (str): _description_

    Returns:
        K: 3x3 intrinsics matrix K
    """
    K = np.zeros((3, 3), dtype=np.float32)
    ydict = {}
    with open(f, "r") as fin:
        ydict = yaml.safe_load(fin)

    K[0, 0] = ydict["sensor"]["intrinsics"][0]
    K[1, 1] = ydict["sensor"]["intrinsics"][1]
    K[0, 2] = ydict["sensor"]["intrinsics"][2]
    K[1, 2] = ydict["sensor"]["intrinsics"][3]
    K[2, 2] = 1
    return K


def read_img_size(f: str) -> Tuple[int, int]:
    ydict = {}
    with open(f, "r") as fin:
        ydict = yaml.safe_load(fin)
    img_rows = ydict["sensor"]["resolution"][1]
    img_cols = ydict["sensor"]["resolution"][0]
    return img_rows, img_cols


def read_intrinsics_txt(f: str) -> Tuple[np.float32]:
    """Returns camera intrinsics

    Args:
        f (str): _description_

    Returns:
        camera_matrix: 3x3 intrinsics matrix K
        dist_coeffs: 1xM distortion coefficient vector
    """
    with open(f, "r") as file:
        K = np.zeros((3, 3), dtype=np.float32)
        dist_coeffs = 0
        for line in file:
            if line.startswith("P_rect_00"):
                parts = line.split()
                values = [float(value) for value in parts[1:]]
                P = np.array(values).reshape(3, 4)
                K = P[:3, :3]
                K /= K[2, 2]
            if line.startswith("D_00"):
                parts = line.split()
                dist_coeffs = [float(value) for value in parts[1:]]
        return K, dist_coeffs


def read_img_size_txt(f: str) -> Tuple[np.float32]:

    with open(f, "r") as file:
        for line in file:
            if line.startswith("S_rect_00"):
                parts = line.split()
                return int(float(parts[1])), int(float(parts[2]))
    return None

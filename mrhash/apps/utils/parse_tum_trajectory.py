import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_line(line: str):
    parts = line.split()
    timestamp = float(parts[0])
    tx, ty, tz = map(float, parts[1:4])
    qx, qy, qz, qw = map(float, parts[4:8])
    return timestamp, tx, ty, tz, qx, qy, qz, qw


def quaternion_to_matrix(
    qx: np.float32, qy: np.float32, qz: np.float32, qw: np.float32
):
    # create a rotation object from the quaternion
    r = R.from_quat([qx, qy, qz, qw])
    # convert the rotation object to a rotation matrix
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def construct_homogeneous_matrix(
    tx: np.float32, ty: np.float32, tz: np.float32, rotation_matrix: np.array
):
    # construct the 4x4 homogeneous matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = [tx, ty, tz]
    return transformation_matrix


def parse_TUM_trajectory(file_path: str):
    poses, quaternions, timestamps = list(), list(), list()
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            timestamp, tx, ty, tz, qx, qy, qz, qw = parse_line(line)
            quaternions.append(np.array([qx, qy, qz, qw], dtype=np.float32))
            poses.append(np.array([tx, ty, tz], dtype=np.float32))
            # rotation_matrix = quaternion_to_matrix(qx, qy, qz, qw)
            # homogeneous_matrix = construct_homogeneous_matrix(tx, ty, tz, rotation_matrix)
            # homogeneous_matrices.append(homogeneous_matrix)
            timestamps.append(timestamp)
    return poses, quaternions, timestamps


def parse_KITTI_line(line: str):
    parts = line.split()
    rotation_matrix = np.array(
        [
            [float(part) for part in parts[0:3]],
            [float(part) for part in parts[4:7]],
            [float(part) for part in parts[8:11]],
        ]
    )
    rotation_matrix = R.from_matrix(rotation_matrix)
    quat = rotation_matrix.as_quat()
    tx, ty, tz = map(float, [parts[3], parts[7], parts[11]])
    return tx, ty, tz, quat


def parse_KITTI360_line(line: str):
    parts = line.split()
    timestamp = int(parts[0])
    rotation_matrix = np.array(
        [
            [float(part) for part in parts[1:4]],
            [float(part) for part in parts[5:8]],
            [float(part) for part in parts[9:12]],
        ]
    )
    rotation_matrix = R.from_matrix(rotation_matrix)
    quat = rotation_matrix.as_quat()
    tx, ty, tz = map(float, [parts[4], parts[8], parts[12]])
    return timestamp, tx, ty, tz, quat


def parse_KITTI_trajectory(file_path: str):
    poses, quaternions, timestamps = list(), list(), list()
    timestamp = 0
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            tx, ty, tz, quat = parse_KITTI_line(line)
            quaternions.append(quat)
            poses.append(np.array([tx, ty, tz], dtype=np.float32))
            timestamps.append(timestamp)
            timestamp += 1
    return poses, quaternions, timestamps


def parse_KITTI360_trajectory(file_path: str):
    poses, quaternions, timestamps = list(), list(), list()
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                continue
            timestamp, tx, ty, tz, quat = parse_KITTI360_line(line)
            quaternions.append(quat)
            poses.append(np.array([tx, ty, tz], dtype=np.float32))
            timestamps.append(timestamp)
    return poses, quaternions, timestamps


def inv(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = np.transpose(T[:3, :3])
    T_inv[:3, 3] = -np.transpose(T[:3, :3]) @ T[:3, 3]
    return T_inv


def transform_trajectory(translations, quaternions, Tcs):

    size = len(translations)

    rotation_matrices = np.zeros((size, 3, 3))
    for i, quaternion in enumerate(quaternions):
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        rotation_matrices[i] = rotation_matrix

    delta_rot = np.zeros((size, 3, 3))
    delta_t = np.zeros((size, 3))
    delta_T = np.zeros((size, 4, 4))
    for i in range(1, size):
        delta_rot[i - 1] = np.transpose(rotation_matrices[i - 1]) @ rotation_matrices[i]
        delta_t[i - 1] = np.transpose(rotation_matrices[i - 1]) @ (
            translations[i] - translations[i - 1]
        )
        delta_T[i - 1] = np.eye(4)
        delta_T[i - 1, :3, :3] = delta_rot[i - 1]
        delta_T[i - 1, :3, 3] = delta_t[i - 1]

    for i in range(size):
        delta_T[i] = inv(Tcs) @ delta_T[i] @ Tcs

    transformed_poses = np.zeros((size, 4, 4))
    transformed_poses[0] = np.eye(4)
    for i in range(1, size):
        transformed_poses[i] = np.eye(4)
        transformed_poses[i] = transformed_poses[i - 1] @ delta_T[i - 1]

    transformed_positions = transformed_poses[:, :3, 3]
    transformed_quaternions = np.zeros((size, 4))
    for i in range(size):
        rot = R.from_matrix(transformed_poses[i, :3, :3])
        transformed_quaternions[i] = rot.as_quat()

    return transformed_positions.tolist(), transformed_quaternions.tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a file of transformations to homogeneous matrices."
    )
    parser.add_argument("file_path", type=str, help="Path to the input file.")
    args = parser.parse_args()

    homogeneous_matrices, timestamps = parse_TUM_trajectory(args.file_path)
    print(homogeneous_matrices[0], timestamps[0])

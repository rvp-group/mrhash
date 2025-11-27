from pathlib import Path
from typing import Tuple
import natsort
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


class KittiReader:
    def __init__(
        self,
        data_dir: Path,
        min_range=0.01,
        max_range=100,
        transform_pcd=False,
        *args,
        **kwargs,
    ):
        """
        :param data_dir: Directory containing the data
        :param min_range: minimum range for the points
        :param max_range: maximum range for the points
        :param transform_pcd: toggle to get pcd in world frame
        :param args:
        :param kwargs:
        """
        sensor_hz = kwargs.pop("sensor_hz")

        self.data_dir = data_dir
        self.cloud_dir = data_dir.joinpath("velodyne")
        self.file_names = natsort.natsorted(
            [file for file in list(self.cloud_dir.glob("*.bin"))]
        )
        self.gt_poses_list = self.read_gt_poses_file(data_dir.joinpath("poses.txt"))
        self.transform_pcd = transform_pcd
        if len(self.file_names) != len(self.gt_poses_list):
            print("ERROR size mismatch")
        self.min_range = min_range
        self.max_range = max_range
        self.time = 0.0
        self.time_inc = 1.0 / sensor_hz
        self.file_index = 0
        self.data_dir = data_dir
        self.cdtype = np.float32
        if (self.data_dir / ".dtype.pkl").exists():
            f = open(self.data_dir / ".dtype.pkl", "rb")
            self.cdtype = pickle.load(f)

    def read_gt_poses_file(self, filename: Path):
        poses = np.loadtxt(filename, delimiter=" ")
        return poses.reshape((len(poses), 3, 4))

    def __len__(self):
        return len(self.file_names)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __iter__(self):
        self.file_index = 0
        return self

    def __next__(self):
        if self.file_index >= len(self):
            raise StopIteration
        result = self[self.file_index]
        self.file_index += 1
        return result

    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if item >= len(self):
            raise IndexError("Index out of bounds")

        pose = self.gt_poses_list[item]
        rotation_matrix = R.from_matrix(pose[0:3, 0:3])
        quat = rotation_matrix.as_quat()
        translation = pose[0:3, 3]

        cloud_np = np.fromfile(self.file_names[item], dtype=self.cdtype)
        # with open(self.file_names[item], 'rb') as f:
        #     buffer = f.read()
        #     cloud_np = np.frombuffer(buffer, dtype=self.cdtype)
        cloud_np = cloud_np.reshape(-1, 4)[:, :3]
        # if (point.norm() < min_range || point.norm() > max_range || std::isnan(point.x()) || std::isnan(point.y()) ||
        #   std::isnan(point.z()))
        norms = np.linalg.norm(cloud_np, axis=1)
        mask = (norms >= self.min_range) & (norms <= self.max_range)
        # Use the mask to filter the points
        filtered_points = cloud_np[mask]
        self.time += self.time_inc
        return translation, quat, filtered_points

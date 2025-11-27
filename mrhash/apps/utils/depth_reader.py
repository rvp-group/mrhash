from pathlib import Path
import numpy as np
from typing import Tuple
import natsort
from scipy.spatial.transform import Rotation as R
from PIL import Image


class DepthReader:
    def __init__(
        self,
        data_dir: Path,
        min_range=0.01,
        max_range=30,
        depth_scaling=1000.0,
        *args,
        **kwargs,
    ):
        """
        :param data_dir: Directory containing the data
        :param min_range: minimum range for the points
        :param max_range: maximum range for the points
        :param depth_scaling: scaling factor for depth images
        :param args:
        :param kwargs:
        """
        self.data_dir = data_dir
        self.rgb_dir = data_dir.joinpath("results")
        self.depth_dir = data_dir.joinpath("results")
        self.depth_file_names = natsort.natsorted(
            [file for file in list(self.depth_dir.glob("*.png"))]
        )
        self.rgb_file_names = natsort.natsorted(
            [file for file in list(self.rgb_dir.glob("*.jpg"))]
        )
        if len(self.depth_file_names) != len(self.rgb_file_names):
            print(
                f"ERROR: size mismatch depth: {len(self.depth_file_names)} != {len(self.rgb_file_names)}"
            )
            exit(-1)
        self.gt_poses_list = self.read_gt_poses_file(data_dir.joinpath("traj.txt"))
        self.min_range = min_range
        self.max_range = max_range
        self.depth_scaling = depth_scaling
        self.file_index = 0
        self.depth_dtype = np.float32

    def read_gt_poses_file(self, filename: Path):
        poses = np.loadtxt(filename, delimiter=" ")
        return poses.reshape((len(poses), 4, 4))

    def __len__(self):
        return len(self.depth_file_names)

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

    def __getitem__(
        self, item
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if item >= len(self):
            raise IndexError("Index out of bounds")

        pose = self.gt_poses_list[item]
        rotation_matrix = R.from_matrix(pose[0:3, 0:3])
        quat = rotation_matrix.as_quat()
        translation = pose[0:3, 3]

        depth = (
            np.array(Image.open(self.depth_file_names[item]), dtype=np.float32)
            / self.depth_scaling
        )

        rgb = np.array(
            Image.open(self.rgb_file_names[item]).convert("RGB"),
            dtype=np.float32,
        )

        return item + 1, translation, quat, depth, rgb

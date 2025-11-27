from typing import Tuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


def xyz_to_spherical(xyz: np.ndarray) -> np.ndarray:
    sph = np.stack(
        [
            np.arctan2(xyz[:, 1], xyz[:, 0]),
            # np.arccos(xyz[:, 2], np.linalg.norm(xyz, axis=1)),
            np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)),
            np.linalg.norm(xyz, axis=1),
        ],
        axis=1,
    )
    return sph


def spherical_to_xyz(sph: np.ndarray) -> np.ndarray:
    xyz = np.stack(
        [
            np.cos(sph[:, 0]) * np.cos(sph[:, 1]) * sph[:, 2],
            np.sin(sph[:, 0]) * np.cos(sph[:, 1]) * sph[:, 2],
            np.sin(sph[:, 1]) * sph[:, 2],
        ],
        axis=1,
    )
    return xyz


def calculate_spherical_intrinsics(
    points: np.ndarray, image_rows: int, image_cols: int
):
    azel = np.stack(
        (
            np.arctan2(points[:, 1], points[:, 0]),
            np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)),
            np.ones_like(points[:, 1], dtype=np.float32),
        ),
        axis=1,
    )

    # compute dynamic vertical fov
    vertical_fov = np.max(azel[:, 1]) - np.min(azel[:, 1])
    horizontal_fov = np.max(azel[:, 0]) - np.min(azel[:, 0])

    # print("project az max {} az min {} el max {} el min {}".format(np.max(azel[:, 0]), np.min(azel[:, 0]), np.max(azel[:, 1]), np.min(azel[:, 1])))

    fx = -float(image_cols - 1) / horizontal_fov
    fy = -float(image_rows - 1) / vertical_fov
    cx = image_cols / 2
    cy = image_rows / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return K, azel, vertical_fov, horizontal_fov


class CameraModel(int, Enum):
    Pinhole = 0
    Spherical = 1


class Camera:
    def __init__(
        self,
        rows: int,
        cols: int,
        K: np.ndarray,
        min_depth: float = 0.0,
        max_depth: float = np.finfo(np.float64).max,
        model: CameraModel = CameraModel.Pinhole,
    ):
        """ "
        :param rows: number of rows
        :param cols: number of columns
        :param K: camera matrix
        :param min_depth: minimum depth for the points
        :param max_depth: maximum depth for the points
        :param model: camera model
        """

        self.rows_ = rows
        self.cols_ = cols
        self.K_ = K
        self.set_intrinsics_(K)
        self.min_depth_ = min_depth
        self.max_depth_ = max_depth
        self.model_ = model
        # self.camera_in_world_ = camera_in_world

    def set_camera_matrix(self, K: np.ndarray):
        """
        :param K: camera or calibration matrix
        """
        self.K_ = K
        self.set_intrinsics_(K)

    def set_intrinsics_(self, K):
        self.fx_ = K[0][0]
        self.fy_ = K[1][1]
        self.cx_ = K[0][2]
        self.cy_ = K[1][2]
        self.ifx_ = 1 / self.fx_
        self.ify_ = 1 / self.fy_

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param points: points to project
        :return lut: look up table containing projections
        :return valid_mask: valid masks for projections
        :return uv_residual: floating points roundoff during projections
        """
        uv = None  # image projections container
        lut = -np.ones((self.rows_, self.cols_), dtype=np.int64)
        depths, valid_mask = self.get_depth(points)

        if self.model_ == CameraModel.Pinhole:
            # camera coords and homogenous division
            cam_coords = self.K_ @ points.T
            uv = (cam_coords / cam_coords[2, :]).T[:, :2]

        elif self.model_ == CameraModel.Spherical:
            # convert to spherical coordinates
            azel = np.stack(
                (
                    np.arctan2(points[:, 1], points[:, 0]),
                    np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)),
                    np.ones_like(points[:, 1], dtype=np.float32),
                ),
                axis=1,
            )
            # homogenous sensor coordinates
            uv = (self.K_ @ azel.T).T[:, :2]

        # round image coordinates
        ui = np.round(uv[:, 0] + 0.5).astype(np.int32)
        vi = np.round(uv[:, 1] + 0.5).astype(np.int32)
        uvi = np.stack((ui, vi), axis=1)
        # calculate residuals for exact inverse projections
        uv_residual = uvi.astype(np.float32) - uv
        valid_mask &= (
            (uvi[:, 0] >= 0)
            & (uvi[:, 0] < self.cols_)
            & (uvi[:, 1] >= 0)
            & (uvi[:, 1] < self.rows_)
        )

        # z-buffer (range or depth)
        # EG
        uvi_ranges = np.concatenate([uvi, depths[:, None]], axis=-1)
        sorted_indices = np.lexsort(
            (uvi_ranges[:, 2], uvi_ranges[:, 1], uvi_ranges[:, 0])
        )
        sorted_stuff = uvi_ranges[sorted_indices]
        _, first_occ_idx, occ_count = np.unique(
            sorted_stuff[:, 0:2], axis=0, return_index=True, return_counts=True
        )
        mask_count = occ_count > 1
        filtered_first_occ = first_occ_idx[mask_count]
        filtered_count_occ = occ_count[mask_count]
        idx_to_keep = np.ones_like(valid_mask, bool)
        for idx, count in zip(filtered_first_occ, filtered_count_occ):
            idx_to_keep[(idx + 1) : (idx + count)] = False
        valid_mask[sorted_indices] = valid_mask[sorted_indices] & idx_to_keep
        # EG

        point_indices = np.arange(points.shape[0], dtype=np.int32)
        valid_uvi = uvi[valid_mask]
        valid_indices = point_indices[valid_mask]

        lut[valid_uvi[:, 1], valid_uvi[:, 0]] = valid_indices

        return lut, valid_mask, uv_residual

    def inverse_project(self, d_img: np.ndarray) -> np.ndarray:
        """
        :param d_img: depth or range image to compute cloud
        :return xyz: point cloud
        """
        # generate pixel coordinates (u, v)
        u = np.arange(self.cols_)
        v = np.arange(self.rows_)
        u, v = np.meshgrid(u, v)
        uv = np.vstack([u.ravel(), v.ravel()]).astype(np.float32)
        x = self.ifx_ * (uv[0, :] - self.cx_ - 0.5)
        y = self.ify_ * (uv[1, :] - self.cy_ - 0.5)
        if self.model_ == CameraModel.Pinhole:
            z = np.ones_like(x)
            xyz = (d_img.flatten().T * np.vstack([x, y, z])).T
            return xyz
        elif self.model_ == CameraModel.Spherical:
            # spherical coordinates azimuth, elevation, range
            spc = np.vstack([x, y, d_img.flatten()]).T
            # to cartesian coordinates
            xyz = spherical_to_xyz(spc)
            return xyz

    def get_depth(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points_d, valid_mask = (
            None,
            None,
        )  # mask for valid points, storage for depth or ranges
        if self.model_ == CameraModel.Pinhole:
            points_d = points[:, 2]
        elif self.model_ == CameraModel.Spherical:
            points_d = np.linalg.norm(points, axis=1)
        valid_mask = (points_d > self.min_depth_) & (points_d < self.max_depth_)
        return points_d, valid_mask


def create_test_depth_image(img_rows, img_cols, some_depth):
    dimage = np.ones((img_rows, img_cols)) * some_depth
    dimage[:, 0::256] += 2
    dimage[0::16, :] += 2
    dimage[10:120, 10:1015] += 2
    dimage[50:70, 500:530] += 2
    dimage[64, 512] -= 30
    return dimage


MAGIC_INTENSITY_NORMALIZER = 5e3


class ImageVisualizer:
    def __init__(
        self,
        num_images: int,
        rows: int,
        cols: int,
        normalizers: list,
        titles: list,
        dtype=np.uint8,
    ):
        plt.ion()
        self.normalizers = normalizers
        self.titles = titles
        fig, axes = plt.subplots(num_images, 1)
        self.figure = fig
        self.axes = axes
        self.axims = list()

        for ax, vmax, title in zip(self.axes, self.normalizers, self.titles):
            array = np.zeros(shape=(rows, cols), dtype=dtype)
            ax.set_xlabel("cols")
            ax.set_ylabel("rows")
            ax.set_title(title)
            self.axims.append(ax.imshow(array, vmin=0, vmax=vmax, cmap="gray"))

    def viz(self, images: list):
        for axim, img in zip(self.axims, images):
            axim.set_data(img)
            self.figure.canvas.flush_events()


if __name__ == "__main__":

    # some fix values for tests
    img_rows = 128
    img_cols = 1024
    fixed_depth = 100
    max_range = 120
    dimage = create_test_depth_image(img_rows, img_cols, fixed_depth)

    # project pinhole
    K = np.array(
        [[400, 0, img_cols / 2], [0, 400, img_rows / 2], [0, 0, 1]], dtype=np.float32
    )

    pinhole = Camera(img_rows, img_cols, K, model=CameraModel.Pinhole)
    point_cloud = pinhole.inverse_project(dimage)
    lut, valid_mask, valid_uv_residual = pinhole.project(point_cloud)
    new_dimage = np.take(dimage, lut)
    if not np.allclose(dimage, new_dimage, atol=1e-7):
        print("depth images are not approximately equals")

    # project spherical
    hfov_max = 2 * np.pi
    hfov_min = 0

    vfov_max = np.pi / 4
    vfov_min = -np.pi / 4

    azres = (hfov_max - hfov_min) / (img_cols + 1)  # compensate for 0-360 wrap
    elres = (vfov_max - vfov_min) / (img_rows)

    K = np.array(
        [[-1 / azres, 0, img_cols / 2], [0, -1 / elres, img_rows / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    spherical = Camera(img_rows, img_cols, K, model=CameraModel.Spherical)

    point_cloud = spherical.inverse_project(dimage)
    lut, valid_mask, valid_uv_residual = spherical.project(point_cloud)
    new_range_img = np.take(dimage, lut)

    new_K, _, vfov, hfov = calculate_spherical_intrinsics(
        point_cloud, img_rows, img_cols
    )

    assert np.allclose(new_K, K, atol=1e-7)
    if not np.allclose(dimage, new_range_img, atol=1e-7):
        print("range images are not approximately equals")
        exit(0)

    print("test run successfully!")

    # dimage = dimage[1:, 1:]
    # new_dimage = new_range_img[1:, 1:]

    ##################### viz ####################
    # viz original range image
    # normalize range values to the range [0, 255]
    # normalized_range = (dimage / max_range) * 255
    # new_normalized_range = (new_range_img / max_range) * 255

    # # convert to unsigned 8-bit integer type
    # dimage_uint8 = normalized_range.astype(np.uint8)
    # new_dimage_uint8 = new_normalized_range.astype(np.uint8)

    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    # axs[0].imshow(dimage_uint8, cmap='gray')
    # axs[0].set_title('gt range image')
    # axs[0].set_xlabel('cols')
    # axs[0].set_ylabel('rows')
    # # fig.colorbar(axs[0].imshow(dimage_uint8, cmap='gray'), ax=axs[0], label='range (scaled)')
    # axs[1].imshow(new_dimage_uint8, cmap='gray')
    # axs[1].set_title('reconstructed range image')
    # axs[1].set_xlabel('cols')
    # axs[1].set_ylabel('rows')
    # # fig.colorbar(axs[1].imshow(new_dimage_uint8, cmap='gray'), ax=axs[1], label='range (scaled)')
    # # residual image

    # residual_img = np.abs(dimage - new_range_img)
    # print("max res", np.max(residual_img))
    # print("argmax res", np.argmax(residual_img))

    # # print(dimage[0, 0])
    # # print(new_range_img[0, 0])
    # # print(point_cloud[0])

    # axs[2].imshow(residual_img, cmap='gray')
    # axs[2].set_title('residual image - avg res ' + str(np.average(residual_img)))
    # axs[2].set_xlabel('cols')
    # axs[2].set_ylabel('rows')
    # fig.colorbar(axs[2].imshow(residual_img, cmap='gray'), ax=axs[2], label='residual')

    # plt.tight_layout()
    # plt.show()

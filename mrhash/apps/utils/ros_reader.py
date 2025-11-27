import bisect
import os
import sys
from pathlib import Path
from typing import Tuple

import natsort
import numpy as np

from utils.point_cloud2 import read_point_cloud


class Ros1Reader:
    def __init__(self, data_dir: Path, min_range=0.01, max_range=100, *args, **kwargs):
        """
        :param data_dir: Directory containing rosbags or path to a rosbag file
        :param min_range: minimum range for the points
        :param max_range: maximum range for the points
        :param args:
        :param kwargs:
        """
        topic = kwargs.pop("topic")

        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print("Rosbags library not installed, run 'pip install -U rosbags'")
            sys.exit(-1)

        self.gt_poses_dict = self.read_gt_poses_file(
            data_dir.joinpath("colosseo_train0_gt.txt")
        )
        self.gt_keys = np.array(sorted(self.gt_poses_dict.keys()))

        if data_dir.is_file():
            self.sequence_id = os.path.basename(data_dir).split(".")[0]
            print(f"setting sequence_id to: {self.sequence_id}")
            self.bag = AnyReader([data_dir])
        else:
            self.sequence_id = os.path.basename(data_dir)[0].split(".")[0]
            self.bag = AnyReader(
                natsort.natsorted([bag for bag in list(data_dir.glob("*.bag"))])
            )
        self.bag.open()

        connection = self.bag.connections

        if not topic:
            raise Exception("You have to specify a topic")

        connection = [x for x in self.bag.connections if x.topic == topic]
        self.msgs = self.bag.messages(connections=connection)
        self.skip = 0

        self.min_range = min_range
        self.max_range = max_range
        self.topic = topic
        self.num_messages = sum(
            reader.topics[topic].msgcount
            for reader in self.bag.readers
            if topic in reader.topics
        )

        if len(self.gt_poses_dict) != self.num_messages:
            print(
                f"WARNING size mismatch | gt poses: {len(self.gt_poses_dict)} != num_messages: {self.num_messages}"
            )

    def read_gt_poses_file(self, filename: Path):
        print("Reading gt poses file")

        raw = np.loadtxt(filename, comments="#", dtype=str)

        ts_str = raw[:, 0]
        rest = raw[:, 1:].astype(float)

        cloud_stamps = []
        for s in ts_str:
            sec_str, nsec_str = s.split(".")
            nsec_str = (nsec_str + "000000000")[:9]
            sec = int(sec_str)
            nsec = int(nsec_str)
            cloud_stamps.append(sec * 1_000_000_000 + nsec)

        cloud_stamps = np.array(cloud_stamps, dtype=np.int64)

        poses = {ts: row for ts, row in zip(cloud_stamps, rest)}

        self.gt_keys = np.sort(cloud_stamps)

        return poses

    def __len__(self):
        return self.num_messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "bag"):
            self.bag.close()

    def nearest_ts(self, ts):
        keys = self.gt_keys
        idx = bisect.bisect_left(keys, ts)
        if idx == 0:
            return keys[0]
        if idx == len(keys):
            return keys[-1]
        prev = keys[idx - 1]
        next_ = keys[idx]
        return prev if abs(ts - prev) < abs(next_ - ts) else next_

    def __iter__(self):
        connection = [x for x in self.bag.connections if x.topic == self.topic]
        self.msgs = self.bag.messages(connections=connection)
        self.skip = 0
        return self

    def __next__(self):
        while True:
            if self.skip > 100:
                print(f"WARNING | already skipped {self.skip} frames.")

            try:
                connection, timestamp, rawdata = next(self.msgs)
            except StopIteration:
                raise StopIteration

            nearest = self.nearest_ts(timestamp)
            if abs(timestamp - nearest) > 1000:
                self.skip += 1
                print(
                    f"WARNING | skipping {timestamp} because nearest ts {nearest} diff: {abs(timestamp-nearest)}"
                )
                continue

            pose = self.gt_poses_dict[nearest]
            t = pose[:3]
            quat = pose[3:]
            msg = self.bag.deserialize(rawdata, connection.msgtype)
            points, _ = read_point_cloud(
                msg, min_range=self.min_range, max_range=self.max_range
            )
            return timestamp, points, t, quat

    def __getitem__(self, item) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        # Create a temporary reader to reach the item (inefficient but correct random access)
        connection = [x for x in self.bag.connections if x.topic == self.topic]
        temp_msgs = self.bag.messages(connections=connection)

        count = 0
        for connection, timestamp, rawdata in temp_msgs:
            nearest = self.nearest_ts(timestamp)
            if abs(timestamp - nearest) > 1000:
                continue

            if count == item:
                pose = self.gt_poses_dict[nearest]
                t = pose[:3]
                quat = pose[3:]
                msg = self.bag.deserialize(rawdata, connection.msgtype)
                points, _ = read_point_cloud(
                    msg, min_range=self.min_range, max_range=self.max_range
                )
                return timestamp, points, t, quat
            count += 1

        raise IndexError("Index out of bounds")

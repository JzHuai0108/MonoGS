import csv
import glob
import os

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass

def pose_matrix_from_pq(pq):
    """ get the 4x4 pose matrix from (p, q) """
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pq[3:]).as_matrix()
    pose[:3, 3] = pq[:3]
    return pose

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.poses = [] # poses of the world frame relative to the camera frame
        self.world_poses = [] # poses of the camera frame relative to the world frame
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        self.world_poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.world_poses.append(pose)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.poses = []
        self.world_poses = []
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = pose_matrix_from_pq(pose_vecs[k][1:8])
            self.world_poses.append(T)
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class RRXIOParser:
    def __init__(self, input_folder, modality, max_dt):
        self.input_folder = input_folder
        self.modality = modality
        self.poses = []
        self.world_poses = []
        self.load_poses(self.input_folder, max_dt, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, max_dt, frame_rate=-1):
        if 'thermal' in self.modality.lower():
            if os.path.isfile(os.path.join(self.input_folder, 'gt_thermal.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_thermal.txt')
            image_list = os.path.join(self.input_folder, 'thermal.txt')
            depth_list = os.path.join(self.input_folder, 'radart.txt')
        elif ('visual' in self.modality.lower()) or ('RGB' in self.modality):
            if os.path.isfile(os.path.join(self.input_folder, 'gt_visual.txt')):
                pose_list = os.path.join(self.input_folder, 'gt_visual.txt')
            image_list = os.path.join(self.input_folder, 'visual.txt')
            depth_list = os.path.join(self.input_folder, 'radarv.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
        print('Found {} associations out of {} images, {} depth images and {} poses'.format(
                len(associations), len(tstamp_image), len(tstamp_depth), len(tstamp_pose)))

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            T = pose_matrix_from_pq(pose_vecs[k][1:8])
            self.world_poses.append(T)
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class SmokeBasementParser:
    def __init__(self, input_folder, modality, max_dt):
        self.input_folder = input_folder
        self.modality = modality
        self.poses = []
        self.world_poses = []
        self.load_poses(self.input_folder, max_dt, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, delimiter=',', skiprows=0):
        data = np.loadtxt(filepath, delimiter=delimiter, dtype=np.unicode_, skiprows=skiprows)
        return data

    def linear_interpol(self, pose_data, time):
        """
        Interpolates pose data at a given time.

        Parameters:
        - pose_data: np.ndarray of shape (N, 8), ordered by time, each row is [time, px, py, pz, qx, qy, qz, qw]
        - time: float, the specific time at which to interpolate the pose

        Returns:
        - interpolated_pose: np.ndarray of shape (4, 4), the interpolated 4x4 pose matrix at the given time
        """
        # Extract times and pose components
        times = pose_data[:, 0]
        poses = pose_data[:, 1:]

        interpolated_translation = np.array([np.interp(time, times, poses[:, i]) for i in range(3)])
        quaternions = Rotation.from_quat(poses[:, 3:7])
        if time <= times[0]:
            interpolated_rotation = quaternions[0]
        elif time >= times[-1]:
            interpolated_rotation = quaternions[-1]
        else:
            slerp = Slerp(times, quaternions)
            interpolated_rotation = slerp(time)

        rotation_matrix = interpolated_rotation.as_matrix()
        interpolated_pose = np.eye(4)
        interpolated_pose[0:3, 3] = interpolated_translation
        interpolated_pose[0:3, 0:3] = rotation_matrix
        return interpolated_pose


    def load_poses(self, datapath, max_dt, frame_rate=-1):
        reldir = 'left/image'
        reltemperdir = 'left/temperature'
        pose_file = os.path.join(self.input_folder, 'kissicp_poses.txt')
        lidar_T_sensor = np.eye(4)
        if 'left' in self.modality.lower():
            image_list = os.path.join(self.input_folder, 'left/times.txt')
            lidar_T_sensor = np.array([[-1, 0, 0, 0.06],
                                        [0, 0, -1, -0.072],
                                        [0, -1, 0, -0.145],
                                        [0, 0, 0, 1]])        
        else:
            reldir = 'right/image'
            reltemperdir = 'right/temperature'
            image_list = os.path.join(self.input_folder, 'right/times.txt')
            lidar_T_sensor = np.array([[-1, 0, 0, -0.06],
                                       [0, 0, -1, -0.072],
                                       [0, -1, 0, -0.145],
                                       [0, 0, 0, 1]])
        pose_data = self.parse_list(pose_file, ' ', 0)
        pose_vals = []
        for data in pose_data:
            pose_row = [float(v) for v in data]
            pose_vals.append(pose_row)
        pose_data = np.array(pose_vals)

        image_data = self.parse_list(image_list)
        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for timepair in image_data:
            color_path = os.path.join(datapath, reldir, timepair[0] + '.png')
            self.color_paths += [color_path]
            if not os.path.isfile(color_path):
                print('Warn: Image file {color_path} does not exist!')
            self.depth_paths += [os.path.join(datapath, reltemperdir, timepair[0] + '.png')] # placeholder for viz

            T = self.linear_interpol(pose_data, float(timepair[0]))
            T = np.dot(T, lidar_T_sensor)
            self.world_poses.append(T)
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(color_path),
                "depth_path": str(color_path),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class VIVIDPPParser:
    """
    The parser for the VIVID++ dataset processed by Huai.
    The processing script is at https://github.com/JzHuai0108/NeRF-VO/blob/main/scripts/vivid_bag_to_folder.py
    The sequences are undistorted, but rgb and thermal have different number of images,
     and the per-pixel depth for thermal/RGB images are unavailable because we cannot deduce the corresponding relation from VIVID++.
    Undistorted rgb images of resolution 640x480, thermal images of resolution 632x464.
    """
    def __init__(self, input_folder, modality, max_dt):
        self.input_folder = input_folder
        self.modality = modality
        self.poses = []
        self.world_poses = []
        self.load_poses(self.input_folder, max_dt, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))
        return associations

    def load_poses(self, datapath, max_dt, frame_rate=-1):
        if 'thermal' in self.modality.lower():
            pose_list = os.path.join(self.input_folder, 'poses_thermal.txt')
            image_list = os.path.join(self.input_folder, self.modality + '.txt')
            depth_list = os.path.join(self.input_folder, 'Depth.txt')
        elif 'RGB' in self.modality:
            pose_list = os.path.join(self.input_folder, 'poses_RGB.txt')
            image_list = os.path.join(self.input_folder, self.modality + '.txt')
            depth_list = os.path.join(self.input_folder, 'Depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
        print('Found {} associations out of {} images, {} depth images and {} poses'.format(
                len(associations), len(tstamp_image), len(tstamp_depth), len(tstamp_pose)))

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            T = pose_matrix_from_pq(pose_vecs[k][1:8])
            self.world_poses.append(T)
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }
            self.frames.append(frame)


class VIVIDParser:
    """
    parser for the vivid processed sequences processed by Shin.
    downloaded from https://github.com/UkcheolShin/ThermalSfMLearner-MS
    These sequences are undistorted, associated depth/rgb/thermal, have per pixel depth for thermal/RGB images,
    rgb images of resolution 640x480, thermal images of resolution 640x512.
    """
    def __init__(self, input_folder, modality):
        self.input_folder = input_folder
        self.modality = modality
        self.poses = []
        self.world_poses = []
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def get_files(self, folder, ext):
        """
        Retrieve all files in a given folder with the specified extension, sorted by filename.

        Args:
            folder (str): Path to the folder.
            ext (str): File extension to look for (e.g., ".txt", ".png").

        Returns:
            list: Sorted list of file paths with the specified extension.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"The folder '{folder}' does not exist or is not a directory.")
        if not ext.startswith("."):
            ext = f".{ext}"  # Ensure the extension starts with a dot
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(ext.lower())
        ]
        # Sort files naturally (e.g., "0001.npy", "0002.npy")
        files.sort(key=lambda x: os.path.basename(x))
        return files

    def load_poses(self, datapath, frame_rate=-1):
        if 'thermal' in self.modality.lower():
            pose_list = os.path.join(self.input_folder, 'poses_thermal.txt')
            image_dir = os.path.join(self.input_folder, self.modality, 'data')
            depth_dir = os.path.join(self.input_folder, 'Warped_Depth/data_THERMAL')
            if not os.path.isdir(image_dir): # VIVID_256 preprocessed by Shin's bash script
                print('Info: assuming VIVID_256 format')
                image_dir = os.path.join(self.input_folder, self.modality)
                pose_list = os.path.join(self.input_folder, "poses_T.txt")
                depth_dir = os.path.join(self.input_folder, "Depth_T")
        elif 'RGB' in self.modality:
            pose_list = os.path.join(self.input_folder, 'poses_RGB.txt')
            image_dir = os.path.join(self.input_folder, self.modality, 'data')
            depth_dir = os.path.join(self.input_folder, 'Warped_Depth/data_RGB')
            if not os.path.isdir(image_dir): # VIVID_256 preprocessed by Shin's bash script
                print('Info: assuming VIVID_256 format')
                image_dir = os.path.join(self.input_folder, self.modality)
                pose_list = os.path.join(self.input_folder, "poses_RGB.txt")
                depth_dir = os.path.join(self.input_folder, "Depth_RGB")

        image_data = self.get_files(image_dir, '.png')
        depth_data = self.get_files(depth_dir, '.npy')
        pose_data = self.parse_list(pose_list, skiprows=0)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        pose_list = pose_vecs.reshape(-1, 3, 4)

        print('Found {} images, {} depth images, and {} poses'.format(
                len(image_data), len(depth_data), len(pose_list)))

        self.color_paths = image_data
        self.world_poses = [np.vstack((T, [0, 0, 0, 1])) for T in pose_list]
        self.poses = [np.linalg.inv(np.vstack((T, [0, 0, 0, 1]))) for T in pose_list]
        self.depth_paths = depth_data


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.poses = []
        self.world_poses = []
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]
            quat = quat[[1, 2, 3, 0]]
            T_w_i = pose_matrix_from_pq(np.hstack((trans, quat)))
            T_w_c = np.dot(T_w_i, T_i_c0)
            self.world_poses.append(T_w_c)
            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        if 'raw' not in calibration:
            cam0raw = calibration
        else:
            cam0raw = calibration["raw"]
        if 'opt' not in calibration:
            cam0opt = calibration
        else:
            cam0opt = calibration["opt"]
        # Camera prameters
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]
        self.width = cam0opt["width"]
        self.height = cam0opt["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        self.K_raw = np.array(
            [[cam0raw["fx"], 0.0, cam0raw["cx"]], [0.0, cam0raw["fy"], cam0raw["cy"],], [0.0, 0.0, 1.0],]
        )
        # distortion parameters
        self.distorted = cam0raw["distorted"]

        if 'distortion_model' in cam0raw.keys():
            self.distortion_model = cam0raw['distortion_model']
        else:
            self.distortion_model = None
        print(f"Distortion model: {self.distortion_model}")

        if self.distortion_model == 'radtan':
            self.dist_coeffs = np.array(
                [
                    cam0raw["k1"],
                    cam0raw["k2"],
                    cam0raw["p1"],
                    cam0raw["p2"],
                    cam0raw["k3"],
                ]
            )
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K_raw,
                self.dist_coeffs,
                np.eye(3),
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )
        elif self.distortion_model == "equidistant":
            self.dist_coeffs = np.array(
                [
                    cam0raw["k1"],
                    cam0raw["k2"],
                    cam0raw["k3"],
                    cam0raw["k4"]
                ]
            )
            self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                self.K_raw,
                self.dist_coeffs,
                np.eye(3),
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )
        elif self.distortion_model is None or self.distortion_model == "":
            self.dist_coeffs = np.zeros(5)
            self.map1x = None
            self.map1y = None

        # depth parameters
        self.has_depth = True if "depth_scale" in cam0opt.keys() else False
        self.depth_scale = cam0opt["depth_scale"] if self.has_depth else None
        print(f"has depth? {self.has_depth}, depth_scale {self.depth_scale}.")

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        img = cv2.imread(color_path)
        if len(img.shape) == 2 or img.shape[2] == 1:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = img
        image = np.array(rgb_img)
        if self.map1x is not None:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            # cv2.imshow("undistorted", image)
            # cv2.waitKey(3000)

        depth = None
        if self.has_depth:
            depth_path = self.depth_paths[idx]
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = np.array(Image.open(depth_path)) / self.depth_scale
                if self.map1x is not None:
                    depth = cv2.remap(depth, self.map1x, self.map1y, cv2.INTER_NEAREST)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.distorted = calibration["distorted"]
        if 'distortion_model' in calibration.keys():
            self.distortion_model = calibration['distortion_model']
        else:
            self.distortion_model = 'radtan'
        print(f"Distortion model: {self.distortion_model}")

        if self.distortion_model == 'radtan':
            self.dist_coeffs = np.array(
                [
                    cam0raw["k1"],
                    cam0raw["k2"],
                    cam0raw["p1"],
                    cam0raw["p2"],
                    cam0raw["k3"],
                ]
            )
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K_raw,
                self.dist_coeffs,
                self.Rmat,
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )

            self.dist_coeffs_r = np.array(
                [
                    cam1raw["k1"],
                    cam1raw["k2"],
                    cam1raw["p1"],
                    cam1raw["p2"],
                    cam1raw["k3"],
                ]
            )
            self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
                self.K_raw_r,
                self.dist_coeffs_r,
                self.Rmat_r,
                self.K_r,
                (self.width, self.height),
                cv2.CV_32FC1,
            )
        else:
            self.dist_coeffs = np.array(
                [
                    cam0raw["k1"],
                    cam0raw["k2"],
                    cam0raw["k3"],
                    cam0raw["k4"]
                ]
            )
            self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                self.K_raw,
                self.dist_coeffs,
                self.Rmat,
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )

            self.dist_coeffs_r = np.array(
                [
                    cam1raw["k1"],
                    cam1raw["k2"],
                    cam1raw["k3"],
                    cam1raw["k4"]
                ]
            )
            self.map1x_r, self.map1y_r = cv2.fisheye.initUndistortRectifyMap(
                self.K_raw_r,
                self.dist_coeffs_r,
                self.Rmat_r,
                self.K_r,
                (self.width, self.height),
                cv2.CV_32FC1,
            )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.distorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

class RRXIODataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        modality = config['Dataset']['modality']
        parser = RRXIOParser(dataset_path, modality, 0.08)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

class SmokeBasementDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        modality = config['Dataset']['modality']
        parser = SmokeBasementParser(dataset_path, modality, 0.08)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class VIVIDDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        modality = config['Dataset']['modality']
        parser = VIVIDParser(dataset_path, modality)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class VIVIDPPDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        modality = config['Dataset']['modality']
        parser = VIVIDPPParser(dataset_path, modality, max_dt = 0.2)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = EuRoCParser(dataset_path, start_idx=config["Dataset"]["start_idx"])
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.distorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.distorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "rrxio":
        return RRXIODataset(args, path, config)
    elif config["Dataset"]["type"] == "smoke_basement":
        return SmokeBasementDataset(args, path, config)
    elif config["Dataset"]["type"] == "vivid":
        return VIVIDDataset(args, path, config)
    elif config["Dataset"]["type"] == "vividpp":
        return VIVIDPPDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")

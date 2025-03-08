from torch.utils.data import Dataset
import torchvision.transforms as transforms

from nerf.utils.utils import quat2mat

import cv2
import numpy as np


class NeRFDataset(Dataset):

    def __init__(self, cameras, images, image_dir, L_pos=10, L_dir=4, scale=1.0):
        self.cameras = cameras
        self.images = images

        # The scale factor to downsample the images
        self.scale = scale

        self.transform = transforms.ToTensor()

        # The number of positional encodings for the ray origins and directions
        self.L_pos = L_pos
        self.L_dir = L_dir

        # List of image names
        self.image_names = list(images.keys())
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Getting the image data from the COLMAP dataset
        image_name = self.image_names[idx]
        image_data = self.images[image_name]
        camera_id = image_data["camera_id"]
        camera = self.cameras[camera_id]

        # Getting the camera parameters from the COLMAP dataset
        focal = self.scale * camera["params"][0]
        cx = self.scale * camera["params"][1]
        cy = self.scale * camera["params"][2]
        width = int(self.scale * camera["width"])
        height = int(self.scale * camera["height"])

        # Loading the image
        image_path = str(self.image_dir.joinpath(image_name))
        img = cv2.imread(image_path)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0

        qvec = image_data["qvec"].astype("float32")
        tvec = image_data["tvec"].astype("float32")
        R = quat2mat(qvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec

        pose = pose.astype("float32")

        # Ray generation (example - adapt to your NeRF setup)
        rays_o, rays_d = self.generate_rays(pose, width, height, focal, cx, cy)

        return {
            "image": self.transform(img),  # (H, W, 3)
            "rays_o": self.transform(rays_o).squeeze(0),  # (H*W, 3)
            "rays_d": self.transform(rays_d).squeeze(0),  # (H*W, 3)
            "pose": pose,  # (4, 4)
            "focal": focal,
            "cx": cx,
            "cy": cy,
            "image_name": image_name,
        }

    def generate_rays(self, pose, width, height, focal, cx, cy):
        """Generates rays for a given camera pose."""
        # Creates a pixel grid
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
        x = x.astype("float32")
        y = y.astype("float32")

        # Calculate ray directions in camera space
        dirs = np.stack([(x - cx) / focal, -(y - cy) / focal, -np.ones_like(x)], -1)  # (H, W, 3)
        # Rotate ray directions to world space
        rays_d = np.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)  # (H, W, 3)
        # Ray origins are the camera's position in world space
        rays_o = np.broadcast_to(pose[:3, 3], rays_d.shape)  # (H, W, 3)

        rays_d = rays_d.reshape(-1, 3).copy()  # (H*W, 3)
        rays_o = rays_o.reshape(-1, 3).copy()  # (H*W, 3)

        return rays_o, rays_d


class GSDataset(Dataset):

    def __init__(self, cameras, images, image_dir, scale=1.0):
        self.cameras = cameras
        self.images = images

        # The scale factor to downsample the images
        self.scale = scale

        self.transform = transforms.ToTensor()

        # List of image names
        self.image_names = list(images.keys())
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Getting the image data from the COLMAP dataset
        image_name = self.image_names[idx]
        image_data = self.images[image_name]
        camera_id = image_data["camera_id"]
        camera = self.cameras[camera_id]

        # Getting the camera parameters from the COLMAP dataset
        focal = self.scale * camera["params"][0]
        cx = self.scale * camera["params"][1]
        cy = self.scale * camera["params"][2]
        width = int(self.scale * camera["width"])
        height = int(self.scale * camera["height"])

        # Loading the image
        image_path = str(self.image_dir.joinpath(image_name))
        img = cv2.imread(image_path)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0

        qvec = image_data["qvec"].astype("float32")
        tvec = image_data["tvec"].astype("float32")
        R = quat2mat(qvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec

        pose = pose.astype("float32")

        return {
            "image": self.transform(img),  # (H, W, 3)
            "pose": pose,  # (4, 4)
            "focal": focal,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "image_name": image_name,
        }

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from nerf.utils.utils import quat2mat
from nerf.utils.utils import positional_encoding

import cv2
import numpy as np


class NeRFDataset(Dataset):
    def __init__(self, cameras, images, image_dir, L=10):
        self.cameras = cameras
        self.images = images

        self.L = L

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
        focal = camera["params"][0]
        cx = camera["params"][1]
        cy = camera["params"][2]

        # Loading the image
        image_path = str(self.image_dir.joinpath(image_name))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0

        qvec = image_data["qvec"].astype("float32")
        tvec = image_data["tvec"].astype("float32")
        R = quat2mat(qvec)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = tvec

        pose = pose.astype("float32")

        # Ray generation (example - adapt to your NeRF setup)
        rays_o, rays_d = self.generate_rays(pose, camera["width"], camera["height"], focal, cx, cy)

        rays_d = positional_encoding(rays_d, self.L)
        rays_o = positional_encoding(rays_o, self.L)

        return {
            "image": transforms.ToTensor(img),  # (H, W, 3)
            "pose": transforms.ToTensor(pose),  # (4, 4)
            "rays_o": transforms.ToTensor(rays_o),  # (H*W, 3)
            "rays_d": transforms.ToTensor(rays_d),  # (H*W, 3)
            "focal": transforms.ToTensor(focal),
            "cx": transforms.ToTensor(cx),
            "cy": transforms.ToTensor(cy),
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

        rays_d = rays_d.reshape(-1, 3)  # (H*W, 3)
        rays_o = rays_o.reshape(-1, 3)  # (H*W, 3)

        return rays_o, rays_d

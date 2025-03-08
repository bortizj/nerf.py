import torch
import torch.nn as nn


def gaussian_rasterization(
    x, y, cov2d, opacities, features, means_cam, scales, num_gaussians, feature_dim, focal, image_width, image_height, device
):
    image = torch.zeros((image_height, image_width, feature_dim), dtype=torch.float32).to(device)
    alpha_buffer = torch.zeros((image_height, image_width), dtype=torch.float32).to(device)
    depth_buffer = torch.ones((image_height, image_width), dtype=torch.float32).to(device) * float("inf")

    for ii in range(num_gaussians):
        u, v = x[ii], y[ii]
        cov = cov2d[ii]  # Now using the cov variable
        opacity = opacities[ii].item()
        color = features[ii]

        radius = (scales[ii].max() * focal / means_cam[ii, 2]).item() * 3

        min_u = max(0, int(u - radius))
        max_u = min(image_width, int(u + radius) + 1)
        min_v = max(0, int(v - radius))
        max_v = min(image_height, int(v + radius) + 1)

        for py in range(min_v, max_v):
            for px in range(min_u, max_u):
                dx = torch.tensor([px - u.item(), py - v.item()]).to(device)  # vector from center to pixel.
                dist_sq = dx @ torch.inverse(cov) @ dx.T  # Mahalanobis distance

                if dist_sq < 9:  # 3 sigma
                    depth = means_cam[ii, 2].item()
                    alpha = opacity * torch.exp(-0.5 * dist_sq).item()

                    if depth < depth_buffer[py, px]:
                        new_alpha = alpha * (1 - alpha_buffer[py, px])
                        image[py, px] = image[py, px] * (1 - new_alpha) + color * new_alpha
                        alpha_buffer[py, px] = alpha_buffer[py, px] + new_alpha
                        depth_buffer[py, px] = depth

    return image


class GaussianSplattingRenderer(nn.Module):
    def __init__(self, num_gaussians, feature_dim=3):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim

        # Gaussian parameters (learnable)
        self.means = nn.Parameter(torch.randn(num_gaussians, 3))  # 3D means (x, y, z)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3))  # 3D scales (sx, sy, sz)
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))  # Quaternions (w, x, y, z) for rotation
        self.opacities = nn.Parameter(torch.ones(num_gaussians, 1) * -3)  # Logit opacities

        # Color features (learnable)
        self.features = nn.Parameter(torch.randn(num_gaussians, feature_dim))

    def quaternion_to_rotation_matrix(self, quaternions):
        """Convert quaternions to rotation matrices."""
        w, x, y, z = quaternions.unbind(-1)
        two_s = 2.0 / (w * w + x * x + y * y + z * z)
        xx = x * x * two_s
        xy = x * y * two_s
        xz = x * z * two_s
        yw = y * w * two_s
        yy = y * y * two_s
        yz = y * z * two_s
        zw = z * w * two_s
        zz = z * z * two_s
        xw = x * w * two_s

        rot_matrices = torch.stack(
            [1.0 - (yy + zz), xy - zw, xz + yw, xy + zw, 1.0 - (xx + zz), yz - xw, xz - yw, yz + xw, 1.0 - (xx + yy)], -1
        ).reshape(quaternions.shape[:-1] + (3, 3))
        return rot_matrices

    def forward(self, camera_pose, focal, cx, cy, image_width, image_height, device):
        """Render the gaussians."""
        means = self.means
        scales = torch.exp(self.scales)  # Ensure scales are positive
        rotations = self.quaternion_to_rotation_matrix(self.rotations)
        opacities = torch.sigmoid(self.opacities)
        features = torch.sigmoid(self.features)  # Ensure RGB values are between 0 and 1.

        # Transform gaussians to camera space
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        means_cam = (R @ means.T).T + t

        # Project gaussians to image plane
        x = means_cam[:, 0] / means_cam[:, 2] * focal + cx
        y = means_cam[:, 1] / means_cam[:, 2] * focal + cy

        # Compute 2D covariances
        J = torch.tensor([[focal, 0, -cx], [0, focal, -cy], [0, 0, 1]], dtype=torch.float32).to(device)
        J = J @ R
        W = scales
        M = rotations

        V = J @ M @ torch.diag_embed(W)
        cov2d = V[:, :2, :] @ V[:, :2, :].transpose(-1, -2) / means_cam[:, 2].unsqueeze(-1).unsqueeze(-1) ** 2

        image = gaussian_rasterization(
            x,
            y,
            cov2d,
            opacities,
            features,
            means_cam,
            scales,
            self.num_gaussians,
            self.feature_dim,
            focal,
            image_width,
            image_height,
            device,
        )

        return image

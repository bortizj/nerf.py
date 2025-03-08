import numpy as np
import torch
import cv2

import tqdm


def convert_secs_to_hms(seconds):
    """Convert seconds to hours, minutes, and seconds."""
    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"


def render_rays_chunked(model, rays_o, rays_d, near, far, num_samples, L_pos, L_dir, device, chunksize=1024):
    """Renders rays in chunks."""
    batch_size, num_rays, _ = rays_o.shape
    rendered_rgbs = torch.zeros((batch_size, num_rays, 3), device=device)

    if chunksize > num_rays:
        chunksize = num_rays

    for bb in range(batch_size):  # Iterate over batch dimension
        for ii in range(0, num_rays, chunksize):
            rays_o_chunk = rays_o[bb, ii : ii + chunksize]
            rays_d_chunk = rays_d[bb, ii : ii + chunksize]
            rgb_chunk = render_rays(model, rays_o_chunk, rays_d_chunk, near, far, num_samples, L_pos, L_dir, device)
            rendered_rgbs[bb, ii : ii + chunksize] = rgb_chunk

    return rendered_rgbs.reshape(batch_size, num_rays, 3)


def render_rays(model, rays_o, rays_d, near, far, num_samples, L_pos, L_dir, device):
    """Renders rays using volume rendering."""

    # Calculate t values (distances along rays)
    t_vals = torch.linspace(near, far, num_samples).to(device)
    z_vals = near + t_vals * torch.linalg.norm(rays_d, dim=-1, keepdims=True)  # (batch_size, num_samples)

    # Calculate 3D points
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (batch_size, num_samples, 3)

    # Positional encoding
    pts_flat = pts.reshape(-1, 3)  # (batch_size * num_samples, 3)
    pts_encoded = positional_encoding(pts_flat, L_pos)

    # Positional encoding of view directions
    rays_d_flat = rays_d.reshape(-1, 3)
    rays_d_encoded = positional_encoding(rays_d_flat, L_dir)

    # Expand view directions to match points
    rays_d_encoded_expanded = rays_d_encoded[..., None, :].expand(
        pts.shape[:-1] + (rays_d_encoded.shape[-1],)
    )  # (batch_size, num_samples, 27)
    rays_d_encoded_flat = rays_d_encoded_expanded.reshape(-1, rays_d_encoded.shape[-1])

    # Evaluate NeRF network
    rgba = model(pts_encoded, rays_d_encoded_flat)  # (batch_size * num_samples, 4)
    rgba = rgba.reshape(pts.shape[:-1] + (4,))  # (batch_size, num_samples, 4)
    rgb = torch.sigmoid(rgba[..., :3])  # (batch_size, num_samples, 3)
    sigma = torch.relu(rgba[..., 3:])  # (batch_size, num_samples, 1)

    # Volume rendering
    delta_t = t_vals[1] - t_vals[0]
    delta_t = delta_t * torch.ones_like(z_vals)  # (batch_size, num_samples)
    alpha = 1.0 - torch.exp(-sigma * delta_t[..., None])  # (batch_size, num_samples, 1)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-2)  # add small number to avoid zero.
    T = torch.cat([torch.ones_like(T[..., :1, :]), T[..., :-1, :]], dim=-2)  # (batch_size, num_samples, 1)
    weights = alpha * T  # (batch_size, num_samples, 1)

    rgb_rendered = torch.sum(weights * rgb, dim=-2)  # (batch_size, 3)

    return rgb_rendered


def positional_encoding(x, L):
    out = [x]
    for jj in range(L):
        out.append(torch.sin(2**jj * x))
        out.append(torch.cos(2**jj * x))
    return torch.cat(out, dim=1)


def count_parameters(model):
    """Count the number of parameters in a model."""
    total_params = 0
    param_pair = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        param_pair.append([name, params])
        total_params += params

    return total_params, param_pair


def quat2mat(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return R.astype("float32")


def parse_colmap_data(cameras_txt, images_txt, points3D_txt):
    """Parses COLMAP output files and returns data in a structured format."""

    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 141
    cameras = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))  # Convert to floats
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 141, mean observations per image: 1118.8794326241134
    images = {}
    with open(images_txt, "r") as f:
        lines = f.readlines()
        ii = 0
        while ii < len(lines):
            if lines[ii].startswith("#"):
                ii += 1
                continue
            image_parts = lines[ii].split()
            image_id = int(image_parts[0])
            qw, qx, qy, qz, tx, ty, tz, camera_id, name = image_parts[1:]
            qw, qx, qy, qz, tx, ty, tz = map(float, [qw, qx, qy, qz, tx, ty, tz])

            # Use image name as key
            images[name] = {
                "id": image_id,
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": int(camera_id),
                "points2D": [],
            }

            ii += 1
            points2D_line = lines[ii].split()
            num_points = len(points2D_line) // 3
            # Points are in a single line with format (X, Y, POINT3D_ID)
            for jj in range(num_points):
                x, y, point3D_id = map(float, points2D_line[jj * 3 : (jj + 1) * 3])
                images[name]["points2D"].append(
                    {"xy": np.array([x, y]), "point3D_id": int(point3D_id) if int(point3D_id) != -1 else None}
                )  # Handle missing 3D points

            ii += 1

    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    # Number of points: 18013, mean track length: 8.7582301671015372
    points3D = {}
    with open(points3D_txt, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            point3D_id = int(parts[0])
            x, y, z, r, g, b, error = map(float, parts[1:8])
            points3D[point3D_id] = {
                "xyz": np.array([x, y, z]),
                "rgb": np.array([r, g, b]) / 255.0,
                "error": error,
                "tracks": [],
            }  # Normalize RGB values
            # Handle tracks (image_id, point2D_index) if needed.  Not strictly necessary for most NeRF training.
            # You'll need to parse the TRACK[] part of the line if you need it.

    return cameras, images, points3D


def save_predictions(model, loader, folder, device, samples_per_ray=128, l_pos=10, l_dir=4, near=2.0, far=6.0, chunksize=1024):
    """Save the predictions to a file."""
    model.eval()

    tqdm_loop = tqdm.tqdm(loader, ncols=150, desc="Storing")
    with torch.no_grad():
        for ii, data in enumerate(tqdm_loop):
            # The image data from the view
            img = data["image"]
            img_names = data["image_name"]

            # The generated rays
            rays_o = data["rays_o"].to(device=device)
            rays_d = data["rays_d"].to(device=device)

            # The predicted RGB values
            # pred_rgb = render_rays(model, rays_o, rays_d, near, far, samples_per_ray, l_pos, l_dir, device)
            pred_rgb = render_rays_chunked(model, rays_o, rays_d, near, far, samples_per_ray, l_pos, l_dir, device, chunksize)

            pred_rgb = pred_rgb.reshape(img.shape)

            for jj in range(img.shape[0]):
                pred_rgb = np.array(pred_rgb[jj].squeeze(0).to("cpu").permute(1, 2, 0))
                img = np.array(img[jj].squeeze(0).to("cpu").permute(1, 2, 0))

                out_img = np.clip(np.vstack((img, pred_rgb)), 0, 1)
                out_img = (out_img * 255).astype("uint8")

                cv2.imwrite(folder.joinpath(f"{img_names[jj]}.png"), out_img)

            tqdm_loop.set_postfix()

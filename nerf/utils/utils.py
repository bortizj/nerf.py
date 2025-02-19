import numpy as np


def positional_encoding(x, L):
    """Positional encoding for NeRF."""
    out = [x]
    for ii in range(L):
        out.append(np.sin(2.0**ii * x))
        out.append(np.cos(2.0**ii * x))
    return np.cat(out, -1)


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

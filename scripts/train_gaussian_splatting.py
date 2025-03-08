import numpy as np

import tqdm
import torch

from torch.utils.data import DataLoader

from nerf.utils.utils import parse_colmap_data, count_parameters
from nerf.utils.utils import save_predictions_gs
from nerf.utils.utils import convert_secs_to_hms
from nerf.models.basic_gaussian_splatting import GaussianSplattingRenderer
from nerf.utils.datasets import GSDataset

import time

from nerf import REPO_ROOT


# Parameters for the trainer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-4
SCALE = 0.1
NUM_GAUSSIANS = 10000

NUM_EPOCHS = 1000
BATCH_SIZE = 1
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


def train_fun(loader, model, optimizer, loss_fn, scaler, prev_psnr, prev_epoch):
    """
    Will do one epoch of the training
    """
    # Setting the models in training or validation mode
    model.train()

    # Iterating over all the batches
    tqdm_loop = tqdm.tqdm(loader, ncols=100, desc="Batches")
    loss_list = []
    for __, data in enumerate(tqdm_loop):
        # The image data from the view
        img = data["image"].to(device=DEVICE)
        camera_pose = data["pose"].to(device=DEVICE)
        focal = data["focal"].to(device=DEVICE)
        cx = data["cx"].to(device=DEVICE)
        cy = data["cy"].to(device=DEVICE)
        image_width = data["width"].to(device=DEVICE)
        image_height = data["height"].to(device=DEVICE)
        target_bgr = img.reshape(BATCH_SIZE, -1, 3)

        # Forward step
        optimizer.zero_grad()
        with torch.amp.autocast(DEVICE):
            pred_bgr = model(camera_pose, focal, cx, cy, image_width, image_height, DEVICE)

            loss = loss_fn(target_bgr, pred_bgr)

        # Backward step (optimization step)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_list.append(loss.item())
        tqdm_loop.set_postfix(loss=np.mean(loss_list), psnr=prev_psnr, epoch=prev_epoch)

    return loss_list


if __name__ == "__main__":
    # Relevant paths to data
    path_colmap = REPO_ROOT.joinpath("data", "banana_export")
    path_imgs = REPO_ROOT.joinpath("data", "frames", "banana")
    cameras_txt = path_colmap.joinpath("cameras.txt")
    images_txt = path_colmap.joinpath("images.txt")
    points3D_txt = path_colmap.joinpath("points3D.txt")
    path_model = REPO_ROOT.joinpath("data", "banana_model_gs_0.tar")

    epochs_dir = REPO_ROOT.joinpath("data", "epochs")
    params_dir = REPO_ROOT.joinpath("data", "params")
    params_dir.mkdir(parents=True, exist_ok=True)

    # Getting the point cloud data from the COLMAP dataset to a pytorch dataset
    cameras, images, points3D = parse_colmap_data(cameras_txt, images_txt, points3D_txt)
    dataset = GSDataset(cameras, images, path_imgs, scale=SCALE)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    # Defining the model
    model = GaussianSplattingRenderer(NUM_GAUSSIANS).to(DEVICE)

    # Loos function and optimizer for the generator model.
    # Autoscaler avoids gradients flush to zero due to numerical precision
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(DEVICE)

    if LOAD_MODEL:
        checkpoint = torch.load(str(path_model))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Counting the number of parameters in the model
    total_params, param_pair = count_parameters(model)
    for row in param_pair:
        print("| {:<40} | {:>10} |".format(*row))

    print(f"Total parameters: {total_params}")

    tqdm_loop = tqdm.tqdm(range(NUM_EPOCHS), ncols=150, desc="Epochs")

    prev_psnr = 0
    total_time = 0

    psnr_list = []
    times_list = []

    for epoch in tqdm_loop:
        start_time = time.time()

        loss = train_fun(train_loader, model, optimizer, loss_fn, scaler, prev_psnr, epoch - 1)

        mse = np.mean(loss)
        prev_psnr = 20 * np.log10(1.0 / np.sqrt(mse))

        lapse = time.time() - start_time
        total_time += lapse

        psnr_list.append(prev_psnr)
        times_list.append(total_time)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            folder = epochs_dir.joinpath(f"epoch_{epoch + 1}")
            folder.mkdir(parents=True, exist_ok=True)
            save_predictions_gs(model, train_loader, folder, DEVICE)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            state = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, str(params_dir.joinpath(f"epoch_{epoch + 1}.tar")))

        tqdm_loop.set_postfix(e_time=convert_secs_to_hms(lapse), t_time=convert_secs_to_hms(total_time))

    input("Press Enter to finish!")
    print(psnr_list)
    print(times_list)

# Simple NeRF implementation

Repository to develop algorithms for NeRF

## Python environment

### Using Anaconda

Anaconda can be downloaded from the [Anaconda website](https://www.anaconda.com/products/individual).

After installing Anaconda, open a Anaconda prompt and create a new environment. Here we name it pynerf, but the name can be anything.

```shell
conda create --name pynerf python=3.10
# and to activate
conda activate pynerf
# to deactivate
conda deactivate
```

To remove the environment, if you want/need to start fresh.

```shell
conda env remove --name pynerf
```

## Installation

Once you have your environment setup you can install the package and its requirements.

- [Numpy](https://numpy.org/), [opencv](https://opencv.org/), ...
```shell
pip install numpy==1.26.4 opencv-python==4.10.0.84 tqdm packaging psutil
```
- [PyTorch](https://pytorch.org/get-started/locally/)
```shell
# Note that you may need to check the cuda version in your system if available
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```
and
- the package itself
```shell
cd path/to/nerf.py/repo
pip install -e .
```

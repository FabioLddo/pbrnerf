# PBR-NeRF Installation Guide for NVIDIA CUDA Container

This guide provides step-by-step instructions to set up and test PBR-NeRF in the NVIDIA CUDA container `nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04`.

## Prerequisites

- Docker with NVIDIA container runtime
- NVIDIA GPU with compute capability 5.2 or higher
- At least 16GB of GPU memory recommended

## Container Setup
<!-- 
You can build your own Docker image for PBR-NeRF using the provided Dockerfile.

```bash
# or build your own image
docker build -t pbrnerf . -f docker/Dockerfile

docker run --gpus all -it --rm -v $(pwd):/workspace pbrnerf

# setup the environment
source /workspace/setup_env.sh

```

OR create it interactively: -->

Start the container with GPU access:

```bash
# docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04 bash
# docker run --gpus all -it --rm nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 bash

docker run --gpus all -it --rm -v $(pwd):/workspace nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 bash

```

## Quick Start (Automated Installation)
For automated installation, use the provided script:

```bash
# Run automated installation script
cd /workspace
chmod +x scripts/docker_install.sh
./scripts/docker_install.sh

# Test installation
python test_installation.py
```

### Environment Variables

The automated script supports the following optional environment variables:

```bash
# Set CUDA architecture for your GPU (default: 89 for RTX 4090)
export TCNN_CUDA_ARCHITECTURES=89  # RTX 4090: 89, RTX 3090/3080: 86, RTX 2080 Ti: 75, GTX 1080 Ti: 61

# Set Wandb API key for automatic login (optional)
export WANDB_API_KEY="your_wandb_api_key_here"

# Set workspace directory (default: /workspace)
export WORKSPACE_DIR="/workspace"

# Run with custom settings
TCNN_CUDA_ARCHITECTURES=86 WANDB_API_KEY="your_key" ./scripts/docker_install.sh
```

## Manual Installation (Step-by-Step)

If you prefer manual installation or need to troubleshoot, follow these detailed steps:

### 1. Update System and Install Dependencies

```bash
# Update package lists
apt-get update

# Install essential packages
apt-get install -y \
    wget \
    curl \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev

# Create symlinks for python
ln -sf /usr/bin/python3 /usr/bin/python

apt install gcc-10 g++-10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100


```

### 2. Set Environment Variables

```bash

# Set CUDA paths
# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set CUDA architecture based on your GPU
# For RTX 4090: 89, RTX 3090/3080: 86, RTX 2080 Ti: 75, GTX 1080 Ti: 61
# export TCNN_CUDA_ARCHITECTURES=89  # Adjust based on your GPU

# Enable OpenEXR support
export OPENCV_IO_ENABLE_OPENEXR=True
```

### 3. Download and Install OptiX SDK

```bash
# Navigate to workspace
cd /workspace

# Download OptiX SDK 7.6.0 (you'll need to download this from NVIDIA's website with your account)
# For this guide, we assume you have the installer file
# wget https://developer.download.nvidia.com/designworks/optix/secure/7.6.0/linux64-x86_64/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64-31894579.sh

# Make installer executable and run it
# chmod +x NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64-31894579.sh
# ./NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64-31894579.sh --include-subdir --skip-license

wget https://developer.nvidia.com/optix/downloads/7.3.0/linux64

chmod +x NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh

./NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh --include-subdir --skip-license

# Set OptiX path
export OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
```

### 4. Create Python Virtual Environment

```bash
# Create virtual environment
# python3 -m venv pbrnerf_env
# source pbrnerf_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools

```

### 5. Install Python Dependencies

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118


# Install other dependencies

pip install mkl==2023.1.0

pip install --no-input \
    lpips \
    opencv-python \
    open3d \
    tqdm \
    imageio \
    scikit-image \
    scikit-learn \
    trimesh \
    pyexr \
    einops \
    wandb \
    OpenEXR \
    matplotlib \
    numpy \
    pybind11

# Install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch

```

### 6. Set Additional Environment Variables

```bash
# Set Torch directory for CMake
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch

# Add pybind11 path
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### 7. Build PyRT Module

```bash
# Navigate to pyrt directory
cd /workspace/code/pyrt

# Create build directory
mkdir -p build
cd build

# Configure CMake
# cmake .. \
#     -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR \
#     -DTorch_DIR=$Torch_DIR \
#     -Dpybind11_DIR=$pybind11_DIR

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CXXFLAGS="-std=c++17"

# Configure with C++17
cmake .. \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR \
    -DTorch_DIR=$Torch_DIR \
    -Dpybind11_DIR=$pybind11_DIR

# Build the module
make pyrt -j$(nproc)

# Verify build
ls -la *.so  # Should show pyrt.cpython*.so
```

### 8. Test Installation

```bash
# Return to workspace root
cd /workspace

# Test PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test tiny-cuda-nn
python -c "import tinycudann as tcnn; print('tiny-cuda-nn imported successfully')"

# Test pyrt module
cd code/pyrt/build
python -c "import pyrt; print('PyRT imported successfully')"
```

### 9. Download Sample Dataset (Optional)

```bash
# Create datasets directory
mkdir -p /workspace/datasets

# For testing, you can download a small sample scene
# Note: Full datasets are large (several GB each)
# Refer to the main README for full dataset download instructions
```

### 10. Login to Weights & Biases

```bash
# Login to wandb for experiment tracking
wandb login
# Enter your API key when prompted
```

## Running a Test

To verify everything works, you can run a quick test:

```bash
cd /workspace

# Activate environment
source pbrnerf_env/bin/activate

# Set environment variables
export TCNN_CUDA_ARCHITECTURES=89  # Adjust for your GPU
export OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
export OPENCV_IO_ENABLE_OPENEXR=True
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Test with a simple Python script
python -c "
import torch
import tinycudann as tcnn
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('tiny-cuda-nn imported successfully')
"

# Add pyrt to Python path and test
cd code/pyrt/build
python -c "import pyrt; print('PyRT module loaded successfully')"
```

## Troubleshooting

### Common Issues:

1. **CUDA architecture mismatch**: Ensure `TCNN_CUDA_ARCHITECTURES` matches your GPU
2. **OptiX not found**: Verify `OptiX_INSTALL_DIR` path is correct
3. **PyRT build fails**: Check that all dependencies are installed and paths are set correctly
4. **Memory issues**: Ensure your GPU has sufficient memory (16GB+ recommended)

### GPU Compute Capability Reference:

- RTX 4090: 89
- RTX 3090/3080: 86
- RTX 2080 Ti: 75
- GTX 1080 Ti: 61
- GTX 1060: 61

Check your GPU's compute capability at: https://developer.nvidia.com/cuda-gpus

## Next Steps

After successful installation:

1. Download the full datasets as described in the main README
2. Run the provided training scripts
3. Monitor progress with Weights & Biases
4. Evaluate results using the provided evaluation scripts

For detailed usage instructions, refer to the main README.md file.

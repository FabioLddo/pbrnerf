# Docker Build Instructions for PBR-NeRF

This directory contains Docker configurations for building PBR-NeRF environments.

## Available Dockerfiles

### 1. Dockerfile.direct (Recommended)
A complete Dockerfile that builds everything directly without external scripts for better layer caching and reproducibility.

### 2. Dockerfile (Original)
Uses the shell script approach for installation.

## Building with Dockerfile.direct

### Prerequisites

1. **Download OptiX SDK**: You must manually download the OptiX SDK installer due to NVIDIA's licensing requirements:
   - Visit: https://developer.nvidia.com/optix/downloads/7.3.0/linux64
   - Download: `NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64.sh`
   - Place it in the project root directory

### Build Commands

```bash
# Basic build
docker build -t pbrnerf:latest -f docker/Dockerfile.direct .

# Build with custom CUDA architecture for your GPU
docker build -t pbrnerf:latest -f docker/Dockerfile.direct \
  --build-arg TCNN_CUDA_ARCHITECTURES=86 .

# Build for multiple GPU architectures
docker build -t pbrnerf:latest -f docker/Dockerfile.direct \
  --build-arg TCNN_CUDA_ARCHITECTURES="89;86;80;75" .
```

### Running the Container

```bash
# Run with GPU access and volume mount
docker run --gpus all -it --rm \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/outputs:/workspace/outputs \
  pbrnerf:latest

# Run interactively with shell
docker run --gpus all -it --rm \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/outputs:/workspace/outputs \
  pbrnerf:latest bash

# Run with custom environment variables
docker run --gpus all -it --rm \
  -e WANDB_API_KEY="your_wandb_key" \
  -v $(pwd)/datasets:/workspace/datasets \
  pbrnerf:latest
```

## Build Arguments

The `Dockerfile.direct` supports the following build arguments:

- `TCNN_CUDA_ARCHITECTURES`: CUDA compute capabilities (default: 89 for RTX 4090)
- `PYTORCH_VERSION`: PyTorch version (default: 2.4.0)
- `TORCHVISION_VERSION`: TorchVision version (default: 0.19.0)
- `TORCHAUDIO_VERSION`: TorchAudio version (default: 2.4.0)
- `OPTIX_VERSION`: OptiX SDK version (default: 7.3.0)

## GPU Compute Capability Reference

Set `TCNN_CUDA_ARCHITECTURES` based on your GPU:

- RTX 4090: 89
- RTX 3090/3080: 86
- RTX 2080 Ti: 75
- GTX 1080 Ti: 61

## Troubleshooting

### Common Issues:

1. **OptiX SDK not found**: Ensure the OptiX installer is in the build context root
2. **CUDA architecture mismatch**: Use correct `TCNN_CUDA_ARCHITECTURES` for your GPU
3. **Build fails on PyRT**: Check that all source files are properly copied
4. **Memory issues during build**: Increase Docker memory allocation

### Multi-stage Build (Advanced)

For smaller production images, you can create a multi-stage build by modifying the Dockerfile to separate build dependencies from runtime dependencies.

## File Structure

```
docker/
├── Dockerfile.direct      # Direct installation (recommended)
├── Dockerfile            # Script-based installation
└── README_Docker.md      # This file
```

## Next Steps

After successful build:

1. Download datasets to the mounted `datasets` directory
2. Configure Weights & Biases: `wandb login`
3. Run training scripts from the container
4. Access outputs in the mounted `outputs` directory


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

### NOTES:

Interesting discussion about OptiX:
https://forums.developer.nvidia.com/t/libnvoptix-so-1-not-installed-by-the-driver/221704/7


Test docker commands:

```bash

docker run --gpus all --rm -it --entrypoint /bin/bash ghcr.io/fabiolddo/pbrnerf:latest

docker run --gpus all --rm -it --entrypoint /bin/bash -v ./datasets:/workspace/datasets/ -v /usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro -v /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro ghcr.io/fabiolddo/pbrnerf:latest


docker run --gpus all --rm -it --entrypoint /bin/bash -v ./datasets:/workspace/datasets/ -v /usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro -v /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw ghcr.io/fabiolddo/pbrnerf:latest

```

Building optix sample??

```bash

cd $OPTIX_ROOT
mkdir build && cd build
cmake ..
make -j$(nproc)

./optixHello

```

Test:

```bash

# clean and reconfigure the OptiX SDK samples
cd /workspace/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/SDK
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 \
  -DCMAKE_CUDA_ARCHITECTURES=89

# build the samples you need (or 'make -j$(nproc)')
make -j"$(nproc)" optixHello optixPathTracer

./optixHello
./optixPathTracer

# or 

make -k

```

```bash

docker run --gpus all --rm -it --entrypoint /bin/bash ghcr.io/fabiolddo/pbrnerf:latest

docker run --gpus all --rm -it --entrypoint /bin/bash -v ./datasets:/workspace/datasets/ -v /usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro -v /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro ghcr.io/fabiolddo/pbrnerf:latest


docker run --gpus all --rm -it --entrypoint /bin/bash -v ./datasets:/workspace/datasets/ -v /usr/lib/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro -v /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw ghcr.io/fabiolddo/pbrnerf:latest
```




FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Build arguments for flexible configuration
ARG GCC_VERSION=10
ARG GXX_VERSION=10
ARG CXX_STANDARD=17

ARG CUDA_VERSION=11.8
ARG CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
#ARG CUDA_ARCHITECTURES="89"

ARG OPTIX_VERSION=7.6.0

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.4.0
ARG TORCHVISION_VERSION=0.19.0
ARG TORCHAUDIO_VERSION=2.4.0
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.7;8.9;9.0;9.0+PTX"

ARG PIP_TIMEOUT=300
ARG PIP_RETRIES=5

# Workspace path argument
ARG WORKSPACE_DIR=/workspace

# Environment variables synchronized with build args
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_IO_ENABLE_OPENEXR=True

#ENV CC=/usr/bin/gcc-${GCC_VERSION}
#ENV CXX=/usr/bin/g++-${GXX_VERSION}
#ENV CXXFLAGS="-std=c++${CXX_STANDARD}"

ENV OPTIX_VERSION=${OPTIX_VERSION}

ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV TORCHVISION_VERSION=${TORCHVISION_VERSION}
ENV TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# ENV TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}
# ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
# ENV CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}

ENV PIP_DEFAULT_TIMEOUT=${PIP_TIMEOUT}
ENV PIP_RETRIES=${PIP_RETRIES}

ENV WORKSPACE_DIR=${WORKSPACE_DIR}

ENV OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV OPTIX_ROOT=/workspace/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64
ENV PATH=$OPTIX_ROOT/bin:$PATH

#WORKDIR /workspaces/pbrnerf
#WORKDIR /workspace
WORKDIR ${WORKSPACE_DIR}

# Copy source code
COPY . ./

# Step 1: Update system and install dependencies
RUN apt-get update -qq && \
    apt-get install -y -qq \
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
        mesa-common-dev \
        xorg-dev \
        gcc-${GCC_VERSION} \
        g++-${GXX_VERSION} \
        libopenexr-dev \
        zlib1g-dev \
        xorg-dev libglu1-mesa-dev \
        doxygen && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GXX_VERSION} 100 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Upgrade pip and setuptools
#RUN pip install --no-cache-dir --upgrade pip setuptools
#RUN pip install --no-cache-dir --upgrade pip

# Step 3: Install PyTorch with CUDA support
# RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
#     --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}

RUN /bin/bash -c 'pip install --no-cache-dir torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}'

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Install tiny-cuda-nn
#RUN pip3 install --no-cache-dir "pybind11<2.12" ninja
#RUN TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" pip install --no-cache-dir -v ninja git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch
#RUN TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" pip install --no-cache-dir -v ninja git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch

# Alternative: Install tiny-cuda-nn from local files, necessary for Cluster builds
RUN TCNN_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" pip install --no-cache-dir -v ninja tiny-cuda-nn/bindings/torch

# Step 6: Install OptiX SDK
COPY NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh ./
RUN chmod +x ./NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh && \
    ./NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh --include-subdir --skip-license

#ENV OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64
#ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
#ENV OPTIX_ROOT=/workspace/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64
#ENV PATH=$OPTIX_ROOT/bin:$PATH

# Step 7: Set PyTorch and pybind11 directories
# RUN echo "export Torch_DIR=\$(python -c \"import torch; print(torch.utils.cmake_prefix_path)\")/Torch" >> /workspace/setup_env.sh && \
#     echo "export pybind11_DIR=\$(python -c \"import pybind11; print(pybind11.get_cmake_dir())\")" >> /workspace/setup_env.sh && \
#     echo "export OptiX_INSTALL_DIR=/workspace/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64" >> /workspace/setup_env.sh && \
#     echo "export OPENCV_IO_ENABLE_OPENEXR=True" >> /workspace/setup_env.sh && \
#     echo "export CC=/usr/bin/gcc-${GCC_VERSION}" >> /workspace/setup_env.sh && \
#     echo "export CXX=/usr/bin/g++-${GXX_VERSION}" >> /workspace/setup_env.sh && \
#     echo "export CXXFLAGS=\"-std=c++${CXX_STANDARD}\"" >> /workspace/setup_env.sh && \
#     echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH}" >> /workspace/setup_env.sh && \
#     chmod +x /workspace/setup_env.sh

# Step 8: Build PyRT module
RUN cd ./code/pyrt && \
    mkdir -p build && \
    cd build && \
    export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch && \
    export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    cmake .. \
        -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR \
        -DTorch_DIR=$Torch_DIR \
        -Dpybind11_DIR=$pybind11_DIR && \
    make pyrt -j$(nproc)

# Step 8: Link OptiX library
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvoptix.so.575.57.08 /usr/lib/x86_64-linux-gnu/libnvoptix.so.1

# Step 9: Create test installation script

# Final verification
# RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" && \
#     python -c "import tinycudann as tcnn; print('tiny-cuda-nn imported successfully')" && \
#     cd /workspace/code/pyrt/build && python -c "import pyrt; print('PyRT imported successfully')"

# ENTRYPOINT ["cd", "/workspace/code", "&&", "python3", "training/train.py"]


# Copy and set entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set working directory to code for training
WORKDIR ${WORKSPACE_DIR}


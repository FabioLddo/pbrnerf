#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration variables
OPTIX_VERSION="7.3.0"
PYTORCH_VERSION="2.4.0"
TORCHVISION_VERSION="0.19.0"
TORCHAUDIO_VERSION="2.4.0"
# CUDA_ARCHITECTURES="90;89;86;80;75;70;61"
# TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89;86;80;75;70}"  # Default to RTX 4090, can be overridden
CUDA_ARCHITECTURES="89"
TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"  # Default to RTX 4090, can be overridden
# TCNN_CUDA_ARCHITECTURES=""
WANDB_API_KEY="${WANDB_API_KEY:-}"  # Optional wandb API key
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

print_status "Starting PBR-NeRF installation in Docker container..."
print_status "CUDA Architecture: ${TCNN_CUDA_ARCHITECTURES}"
print_status "Workspace Directory: ${WORKSPACE_DIR}"

# Step 1: Update system and install dependencies
print_status "Step 1/10: Updating system and installing dependencies..."
apt-get update -qq

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
    xorg-dev libglu1-mesa-dev \
    gcc-10 \
    g++-10

# Create symlinks for python
ln -sf /usr/bin/python3 /usr/bin/python

# Set up gcc/g++ alternatives
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

print_success "System dependencies installed successfully"

# Step 2: Set environment variables
print_status "Step 2/10: Setting up environment variables..."

# Export environment variables
export OPENCV_IO_ENABLE_OPENEXR=True
export TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CXXFLAGS="-std=c++17"

# Create environment setup script for future use
cat > ${WORKSPACE_DIR}/setup_env.sh << 'EOF'
#!/bin/bash
# PBR-NeRF Environment Setup
export OPENCV_IO_ENABLE_OPENEXR=True
export TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES:-89}
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CXXFLAGS="-std=c++17"
export OptiX_INSTALL_DIR=${WORKSPACE_DIR}/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)/Torch
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)
EOF

chmod +x ${WORKSPACE_DIR}/setup_env.sh
source ${WORKSPACE_DIR}/setup_env.sh

print_success "Environment variables configured"

# Step 3: Download and install OptiX SDK
print_status "Step 3/10: Downloading and installing OptiX SDK..."
cd ${WORKSPACE_DIR}

# if [ ! -f "NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh" ]; then
#     print_warning "OptiX SDK installer not found. Please download it manually from:"
#     print_warning "https://developer.nvidia.com/optix/downloads/7.3.0/linux64"
#     print_warning "Save it as: ${WORKSPACE_DIR}/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh"
    
#     # Try to download (this will likely fail due to authentication requirements)
#     print_status "Attempting to download OptiX SDK (may require manual download)..."
#     wget -q https://developer.nvidia.com/optix/downloads/7.3.0/linux64 -O NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh || true
# fi

if [ -f "NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh" ]; then
    chmod +x NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh
    ./NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64.sh --include-subdir --skip-license
    export OptiX_INSTALL_DIR=${WORKSPACE_DIR}/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64
    print_success "OptiX SDK installed successfully"
else
    print_error "OptiX SDK installer not found. Please download manually and re-run the script."
    exit 1
fi

# Step 4: Upgrade pip
print_status "Step 4/10: Upgrading pip and setuptools..."
pip install --upgrade pip setuptools -q
print_success "Pip upgraded successfully"

# Step 5: Install PyTorch with CUDA support
print_status "Step 5/10: Installing PyTorch with CUDA 11.8 support..."
pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/cu118 -q
print_success "PyTorch installed successfully"

# Step 6: Install other Python dependencies
print_status "Step 6/10: Installing Python dependencies..."
pip install mkl==2023.1.0 -q

pip install --no-input -q \
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

print_success "Python dependencies installed successfully"

# Step 7: Install tiny-cuda-nn
print_status "Step 7/10: Installing tiny-cuda-nn..."
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch -q
print_success "tiny-cuda-nn installed successfully"

# Step 8: Set additional environment variables
print_status "Step 8/10: Setting additional environment variables..."
export Torch_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
export pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# Update environment setup script
cat >> ${WORKSPACE_DIR}/setup_env.sh << EOF
export Torch_DIR=\$(python -c "import torch; print(torch.utils.cmake_prefix_path)")/Torch
export pybind11_DIR=\$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
EOF

print_success "Additional environment variables set"

# Step 9: Build PyRT module
print_status "Step 9/10: Building PyRT module..."

# Check if pyrt directory exists
if [ ! -d "${WORKSPACE_DIR}/code/pyrt" ]; then
    print_error "PyRT source directory not found at ${WORKSPACE_DIR}/code/pyrt"
    print_error "Please ensure the PBR-NeRF source code is mounted in the container"
    exit 1
fi

cd ${WORKSPACE_DIR}/code/pyrt
mkdir -p build
cd build

print_status "Configuring CMake for PyRT..."
cmake .. \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR \
    -DTorch_DIR=$Torch_DIR \
    -Dpybind11_DIR=$pybind11_DIR

print_status "Building PyRT module..."
make pyrt -j$(nproc)

# Verify build
if [ -f *.so ]; then
    print_success "PyRT module built successfully"
    ls -la *.so
else
    print_error "PyRT module build failed - no .so file found"
    exit 1
fi

# Step 10: Test installation
print_status "Step 10/10: Testing installation..."

cd ${WORKSPACE_DIR}

# Test PyTorch CUDA
print_status "Testing PyTorch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Test tiny-cuda-nn
print_status "Testing tiny-cuda-nn..."
python -c "import tinycudann as tcnn; print('tiny-cuda-nn imported successfully')"

# Test pyrt module
print_status "Testing PyRT module..."
cd code/pyrt/build
python -c "import pyrt; print('PyRT imported successfully')"

print_success "All tests passed!"

# Optional: Setup wandb
if [ -n "$WANDB_API_KEY" ]; then
    print_status "Setting up Weights & Biases..."
    echo "$WANDB_API_KEY" | wandb login
    print_success "Wandb configured successfully"
else
    print_warning "WANDB_API_KEY not provided. You can login manually later with: wandb login"
fi

# Create a final test script
cd ${WORKSPACE_DIR}
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify PBR-NeRF installation"""

import sys
import os

def test_imports():
    """Test all required imports"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        
        import tinycudann as tcnn
        print("✓ tiny-cuda-nn imported successfully")
        
        # Test pyrt if we're in the right directory
        if os.path.exists('code/pyrt/build'):
            sys.path.insert(0, 'code/pyrt/build')
            import pyrt
            print("✓ PyRT module imported successfully")
        
        import lpips
        import cv2
        import open3d
        import trimesh
        import pyexr
        import wandb
        print("✓ All Python dependencies imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing PBR-NeRF installation...")
    print("=" * 50)
    
    success = test_imports()
    
    print("=" * 50)
    if success:
        print("🎉 Installation test completed successfully!")
        print("\nTo get started:")
        print("1. Source the environment: source setup_env.sh")
        print("2. Download datasets as described in README.md")
        print("3. Login to wandb: wandb login")
        print("4. Run training scripts")
    else:
        print("❌ Installation test failed. Please check the errors above.")
        sys.exit(1)
EOF

chmod +x test_installation.py

print_success "Installation completed successfully!"
print_status "Created files:"
print_status "  - ${WORKSPACE_DIR}/setup_env.sh (environment setup)"
print_status "  - ${WORKSPACE_DIR}/test_installation.py (installation test)"

print_warning "Next steps:"
print_warning "1. Run: source ${WORKSPACE_DIR}/setup_env.sh"
print_warning "2. Run: python ${WORKSPACE_DIR}/test_installation.py"
print_warning "3. If not done already: wandb login"
print_warning "4. Download datasets as described in README.md"

print_success "🎉 PBR-NeRF installation script completed!"

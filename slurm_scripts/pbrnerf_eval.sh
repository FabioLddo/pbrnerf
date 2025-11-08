#!/bin/bash
#SBATCH --job-name="PBRNeRF-eval"
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpuISIN
#SBATCH --gres=gpu:1
#SBATCH--output=logs/R-%x.%j.log

export MPLCONFIGDIR=/workspace/cache/matplotlib_cache
export TORCH_EXTENSIONS_DIR=/workspace/cache/torch_extensions
export PYTORCH_KERNEL_CACHE_PATH=/workspace/cache/pytorch_kernels
export MPLCONFIGDIR=/workspace/cache/matplotlib
export TORCH_HOME=/workspace/cache/torch
export HF_HOME=/workspace/cache/huggingface
export WANDB_CACHE_DIR=/workspace/cache/wandb
export TCNN_RTC_CACHE_DIR=/workspace/cache/tcnn_rtc_cache

# srun singularity exec --bind ~/scratch/pbrnerf/datasets:/workspace/datasets --bind ~/scratch/pbrnerf/outputs:/workspace/outputs --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind ~/scratch/pbrnerf/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind ~/scratch/pbrnerf/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_neilfpp.sh && ls -lah'

# srun singularity exec --bind ~/scratch/pbrnerf/datasets:/workspace/datasets --bind ~/scratch/pbrnerf/outputs:/workspace/outputs --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_neilfpp.sh  && ls -lah'

# ------------------------------------------------------------------------


# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_pbrnerf_neilfpp.sh'

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && cat ./code/training/train.py && ./sota_dtu.sh'

srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./evaluate.sh'

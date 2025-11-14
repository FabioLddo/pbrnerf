#!/bin/bash
#SBATCH --job-name="PBRNeRF"
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuISIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core

#SBATCH --output=logs/R-%x.%j.log

# --output=logs/R-%x.%j.log
# -o logs/R-%x.%j.out # send stdout to outfile
#-e logs/R-%x.%j.err # send stderr to errfile

# ------------------------------------------------------------------------ new image, new wandb

srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_36dd71c.sif /bin/bash -c 'cd /workspace && ./train_pbrnerf_ours.sh'

# ------------------------------------------------------------------------
# ln -sf /usr/lib/x86_64-linux-gnu/libnvoptix.so.575.57.08 /usr/lib/x86_64-linux-gnu/libnvoptix.so.1

# srun singularity exec --bind ~/scratch/pbrnerf/datasets:/workspace/datasets --bind ~/scratch/pbrnerf/outputs:/workspace/outputs --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind ~/scratch/pbrnerf/x86_64-linux-gnu/libnvoptix.so.1:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind ~/scratch/pbrnerf/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.535.261.03:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_neilfpp.sh && ls -lah'

# srun singularity exec --bind ~/scratch/pbrnerf/datasets:/workspace/datasets --bind ~/scratch/pbrnerf/outputs:/workspace/outputs --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_neilfpp.sh  && ls -lah'

# ------------------------------------------------------------------------

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_pbrnerf_neilfpp.sh'

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && cat ./code/training/train.py && ./sota_dtu.sh'

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && cat ./code/training/train.py && ./evaluate.sh'

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && cat ./code/training/train.py && ./sota_neilfpp.sh'

# srun singularity exec --bind ~/scratch/pbrnerf:/workspace --bind ~/scratch/.cache:/home/fabio.loddo/.cache --bind ~/scratch/.cache:/workspace/cache --bind /usr/lib64/libnvoptix.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvoptix.so.1:ro --bind /usr/lib64/libnvidia-rtcore.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.575.57.08:ro --nv ~/scratch/singularity/pbrnerf_latest.sif /bin/bash -c 'cd /workspace && ./train_pbrnerf_ours.sh'

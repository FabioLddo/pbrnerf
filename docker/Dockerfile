FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

COPY ../ /workspace

RUN chmod +x ./scripts/docker_install.sh && ./scripts/docker_install.sh

CMD [ "python3", "test_installation.py" ]

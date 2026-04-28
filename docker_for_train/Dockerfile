ARG CUDA_VERSION=11.6.2
ARG UBUNTU_VERSION=20.04
ARG PYTHON_VERSION=3.9

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    ninja-build \
    libavutil-dev \
    libffi-dev \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6 \
    libosmesa6-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libxext6 \
    libxrender1 \
    patchelf \
    pkg-config \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --python python${PYTHON_VERSION} ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

COPY requirements ./requirements
COPY setup.py ./
COPY diffusion_policy ./diffusion_policy

# Docker training images do not need OpenCV GUI bindings.
RUN sed -i 's/^opencv-python==4.6.0.66$/opencv-python-headless==4.6.0.66/' requirements/uv-common.txt && grep -v '^-r uv-common.txt$' requirements/uv-sim-cu116.txt > requirements/uv-sim-cu116.docker.txt

# gym==0.21.0 ships legacy metadata rejected by pip>=24.1.
RUN uv pip install -r requirements/uv-pip-bootstrap.txt
RUN python -m pip install --no-build-isolation gym==0.21.0
RUN uv pip install --index-strategy unsafe-best-match --build-constraints requirements/uv-build-constraints.txt -r requirements/uv-common.txt
RUN uv pip install --index-strategy unsafe-best-match --build-constraints requirements/uv-build-constraints.txt -r requirements/uv-sim-cu116.docker.txt
RUN uv pip install .

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    MUJOCO_GL=osmesa \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    git \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libusb-1.0-0 \
    libxext6 \
    libxrender1 \
    patchelf \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && useradd --create-home --shell /bin/bash trainer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv
COPY --chown=trainer:trainer . /workspace

ENV HOME=/home/trainer \
    WANDB_DIR=/workspace/wandb \
    MPLCONFIGDIR=/tmp/matplotlib

USER trainer

CMD ["/bin/bash"]

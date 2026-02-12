#!/bin/bash

APT_DEPS="
    gnupg \
    curl \
    wget \
    software-properties-common \
    lsb-release \
    ca-certificates \
"

SYS_DEPS="   
    cmake \
    build-essential \
    gcc-13 \
    g++-13 \
    iproute2 \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libsoup2.4-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    libglm-dev \
    libglfw3-dev \
    liblz4-dev \
    libzstd-dev \
    libspdlog-dev \
    git \
    clangd \
    busybox \
    can-utils \
    nvidia-jetpack \
"

sudo apt update
sudo apt install -y --no-install-recommends $APT_DEPS

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list

sudo apt update
sudo apt install -y --no-install-recommends $SYS_DEPS
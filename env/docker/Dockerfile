FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDACXX=/usr/local/cuda/bin/nvcc

WORKDIR /tmp
COPY installer.sh .
COPY requirements.txt .

RUN apt-get update && apt-get install sudo bash -y

RUN bash installer.sh
RUN pip3 install -r requirements.txt
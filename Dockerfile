FROM tensorflow/tensorflow:1.15.0-gpu-py3
ENV LANG C.UTF-8

ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --upgrade pip"

RUN GIT_CLONE="git clone --depth 10" \
    && rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update

# Basic tools
RUN $APT_INSTALL \
  apt-utils \
  dialog \
  build-essential \
  ca-certificates \
  wget \
  libssl-dev \
  curl \
  unzip \
  unrar \
  vim

# Python
RUN $APT_INSTALL software-properties-common

# Deep Learning Frameworks
RUN $PIP_INSTALL \
  setuptools \
  h5py \
  keras \
  jupyter

# Project dependencies
RUN $APT_INSTALL libsndfile1 -y \
  && $PIP_INSTALL \
    Pillow \
    matplotlib \

# Config & cleanup
RUN ldconfig \
  && apt-get clean \
  && apt-get autoremove \
  && rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# Copy project files
RUN mkdir logoGAN
COPY ./ /logoGAN/
WORKDIR /logoGAN/
EXPOSE 6006

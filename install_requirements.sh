#!/usr/bin/env bash

set -euo pipefail

echo "****************** Preparing system packages ******************"
sudo apt-get update
sudo apt-get install -y libturbojpeg0-dev ninja-build ffmpeg

echo ""
echo "****************** Creating Python 3.12 environment with uv ******************"
uv python pin 3.12
uv venv --python 3.12
uv pip install -p .venv/bin/python \
  colorama \
  cython \
  easydict \
  einops \
  gdown \
  jpeg4py \
  lmdb \
  matplotlib \
  numpy \
  opencv-python \
  pandas \
  pycocotools \
  pyyaml \
  scipy \
  six \
  tensorboard \
  thop \
  timm \
  torch \
  torchaudio \
  torchvision \
  tqdm \
  yacs \
  onnx \
  onnxruntime \
  onnxsim \
  tikzplotlib \
  git+https://github.com/votchallenge/vot-toolkit-python

echo ""
echo "****************** Installation complete ******************"

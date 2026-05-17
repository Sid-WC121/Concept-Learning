#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p ./dataset/models/inception5h
mkdir -p ./dataset/models/mobilenet_v2_1.0_224

# Download inception model
curl -o inception5h.zip https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip -o inception5h.zip -d ./dataset/models/inception5h
rm inception5h.zip

# Download mobilenet model from Google Drive
curl -L -o mobilenet_v2_1.0_224.tgz "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz"
tar -xzf mobilenet_v2_1.0_224.tgz -C ./dataset/models/mobilenet_v2_1.0_224
rm mobilenet_v2_1.0_224.tgz

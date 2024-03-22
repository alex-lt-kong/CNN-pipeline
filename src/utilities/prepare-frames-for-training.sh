#!/bin/bash

TMP_DIR="/tmp/samples"
#rm $TMP_DIR -r
mkdir $TMP_DIR
VIDEO_DIR="$1"
FILE_BASE_NAME="$2"
# FPS=1 means we want to extract two frames per sec
# To extract 5 frames per sec, set it to 5, not 0.2
FPS=1
ffmpeg -i "${VIDEO_DIR}/${FILE_BASE_NAME}" -r "${FPS}" "${TMP_DIR}/${FILE_BASE_NAME}_%05d.jpg"

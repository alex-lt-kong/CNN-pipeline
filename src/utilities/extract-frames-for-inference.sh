IN_FILE=$1
OUT_FILES="/tmp/sample/video_%05d.jpg"
# Ignore FPS if input file is a gif animation.
FPS=5
# OpenCV and PIL have different resizing algorithms that almost
# always result in different images. Preparing the images in
# desired resolution first could avoid this discrepancy
RESOLUTION="640x360"
IMAGE_QUALITY=4
ffmpeg -i "${IN_FILE}" -qscale:v "${IMAGE_QUALITY}" -r "${FPS}" -s "${RESOLUTION}" "${OUT_FILES}"

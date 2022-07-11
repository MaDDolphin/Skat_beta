# ------------------------- POSE MODEL -------------------------

# Загрузка модели обученной на наборе данных COCO
COCO_POSE_URL="https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel"
COCO_FOLDER="pose/coco/"
wget -c ${COCO_POSE_URL} -P ${COCO_FOLDER}
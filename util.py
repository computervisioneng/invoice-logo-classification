import os

import cv2

from keras_yolo3.yolo import YOLO
from utils import detect_object as detect_object_yolo


YOLO_MODEL = None
YOLO_DIR = os.path.join('.', 'models', 'logo_detection')


def get_yolo_detector(model_weights, confidence_threshold=0.5):

    global YOLO_MODEL

    model_classes = os.path.join(YOLO_DIR, 'class.names')
    anchors_path = os.path.join(YOLO_DIR, "yolo_anchors.txt")

    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO(
                        **{
                            "model_path": model_weights,
                            "anchors_path": anchors_path,
                            "classes_path": model_classes,
                            "score": confidence_threshold,
                            "gpu_num": 0,
                            "model_image_size": (416, 416),
                        }
                    )

    return YOLO_MODEL


def detect_object(YOLO_MODEL, img):

    img_path = './tmp.png'
    cv2.imwrite(img_path, img)

    prediction, image = detect_object_yolo(
        YOLO_MODEL,
        img_path,
        save_img=False,
        save_img_path=None,
        postfix='',
    )

    os.remove(img_path)

    return prediction

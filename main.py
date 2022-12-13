import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2

from util import get_yolo_detector, detect_object


def process_detections(detections):
    if len(detections) == 0:
        return 'unknown'
    elif len(detections) == 1 and detections[0]['score'] < 0.6:
        return 'unknown'
    elif len(detections) == 1:
        return detections[0]['category']
    elif len(detections) > 1 and (len([j for j in detections if j['score'] > 0.6]) > 1 or len([j for j in detections if j['score'] > 0.6]) == 0):
        return 'unknown'
    else:
        return [j for j in detections if j['score'] > 0.6][0]['category']

class_names_path = '/home/phillip/Desktop/todays_tutorial/24_invoice_classifier/code/models/logo_detection/class.names'
with open(class_names_path, 'r') as f:
    class_names = [l[:-1] for l in f.readlines() if len(l) > 1]
    f.close()

model_weights = '/home/phillip/Desktop/todays_tutorial/24_invoice_classifier/code/models/logo_detection/' \
                'logo_detector.h5'

detector = get_yolo_detector(model_weights)

test_img_dir = '/home/phillip/Desktop/todays_tutorial/24_invoice_classifier/code/data/test_data'
for test_img_path_ in sorted(os.listdir(test_img_dir)):
    # load invoice
    test_img_path = os.path.join(test_img_dir, test_img_path_)

    img = cv2.imread(test_img_path)

    # detect logo(s)

    detections_ = detect_object(detector, img)

    detections = []
    for detection in detections_:
        _, _, _, _, class_id, confidence_score = detection

        detections.append({'category': class_names[class_id], 'score': confidence_score})

    # process detections
    # return invoice category
    print(test_img_path_, process_detections(detections))

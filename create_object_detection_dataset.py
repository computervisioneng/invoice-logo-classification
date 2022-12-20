import os
import random
import shutil

import cv2
import numpy as np


def get_bbox(mask):
    gry = mask[:, :, 3]
    # blur = cv2.GaussianBlur(gry, (3, 3), 0)
    th = cv2.threshold(gry, 128, 255, cv2.THRESH_BINARY)[1]
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)

    # _ = cv2.rectangle(mask, (int(x - (w / 2)), int(y - (h / 2))), (int(x + (w / 2)), int(y + (h / 2))), (0, 255, 0), 5)
    # _ = cv2.rectangle(mask[:, :, :3].astype(np.int32), (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 5)


    return int(x + (w / 2)), int(y + (h / 2)), int(w), int(h)


def overlay_img(background, overlay, location):

    x, y = location

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x > background_width or y > background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def rotate(img, alpha):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), alpha, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


background_dir = '/media/veracrypt2/TableBank/Detection/images'
logos_dir = '/home/phillip/Desktop/todays_tutorial/24_invoice_classifier/code/data/logos_reduced'

output_dir = '/home/phillip/Desktop/todays_tutorial/24_invoice_classifier/code/data/train_data'

train_dir = os.path.join(output_dir, 'train')
imgs_dir = os.path.join(train_dir, 'imgs')
anns_dir = os.path.join(train_dir, 'anns')

for dir_ in [output_dir, train_dir, imgs_dir, anns_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

with open(os.path.join(output_dir, 'class.names'), 'w') as f:
    for logo_path in sorted(os.listdir(logos_dir)):
        f.write('{}\n'.format(logo_path[:-4]))
    f.close()

with open(os.path.join(output_dir, 'class.names'), 'r') as f:
    classes = [l[:-1] for l in f.readlines() if len(l) > 1]
    f.close()

dataset_size = 10000

for j in range(dataset_size):
    # alpha = random.randint(0, 0)
    background_path = os.path.join(background_dir,
                                   os.listdir(background_dir)[random.randint(0, len(os.listdir(background_dir)) - 1)])
    background_ = cv2.imread(background_path)
    background_ = cv2.resize(background_, (600, 1000))
    background = np.ones((background_.shape[0], background_.shape[1], 4), dtype=np.uint8) * 255
    background[:, :, :3] = background_

    logo_path = os.listdir(logos_dir)[random.randint(0, len(os.listdir(logos_dir)) - 1)]
    logo_img_path = os.path.join(logos_dir, logo_path)

    logo_img = cv2.imread(logo_img_path, -1)

    resize_ = 100 * random.randint(2, 3)
    doc_img_ = cv2.resize(logo_img, (resize_, int(resize_ * logo_img.shape[0] / logo_img.shape[1])))

    print(logo_img_path, doc_img_.shape)
    xc, yc, w, h = get_bbox(doc_img_)

    doc_img_ = doc_img_[int(yc - (h /2)):int(yc + (h /2)), int(xc - (w /2)):int(xc + (w /2)), :]

    try:
        location_x, location_y = random.randint(0, background.shape[1] - w), random.randint(0, background.shape[0] - h)
        # print(location_x, int(w / 2), background.shape[1] - int(w / 2))
        # print(location_y, int(h / 2), background.shape[0] - int(h / 2))
        img_ = overlay_img(background, doc_img_, (location_x, location_y))
        cv2.imwrite(os.path.join(imgs_dir, '{}.jpg'.format(str(j))), img_)

        H, W, _ = img_.shape

        with open(os.path.join(anns_dir, '{}.txt'.format(str(j))), 'w') as f:
            f.write('{} {} {} {} {}\n'.format(classes.index(logo_path[:-4]), str(((w / 2) + location_x) / W),
                                              str(((h / 2) + location_y) / H), str(w / W), str(h / H)))
            f.close()

    except Exception:
        pass

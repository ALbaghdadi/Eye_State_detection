import cv2
import numpy as np
import random
import os
open_eyes = []
closed_eyes = []
def augment_image(image):
    aug_images = [image]

    # Flip
    aug_images.append(cv2.flip(image, 1))

    # Rotate ±15 degrees
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((12, 12), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (24, 24))
        aug_images.append(rotated)

    # Brightness
    factor = random.uniform(0.7, 1.3)
    bright = np.clip(image * factor, 0, 255).astype(np.uint8)
    aug_images.append(bright)

    return aug_images



def append_open(eyes):
    i = 0
    for img in os.listdir(eyes):
        image_path = os.path.join(eyes, img)
        image = cv2.imread(image_path)
        if image is not None and image.shape != (24, 24, 3) :
            image = cv2.resize(image , (24,24))

        if image is not None and image.shape == (24, 24, 3) :
            augmented = augment_image(image)
            open_eyes.extend(augmented)
            i += 1
    print("Open Eyes Loaded (with augmentation):", i, " → Total:", len(open_eyes))

    return open_eyes

def append_closed(eyes):
    i = 0
    for img in os.listdir(eyes):
        image_path = os.path.join(eyes, img)
        image = cv2.imread(image_path)
        if image is not None and image.shape != (24, 24, 3) :
            image = cv2.resize(image , (24,24))
        if image is not None and image.shape == (24, 24, 3):
            augmented = augment_image(image)
            closed_eyes.extend(augmented)
            i += 1
    print("Closed Eyes Loaded (with augmentation):", i, " → Total:", len(closed_eyes))
    return closed_eyes


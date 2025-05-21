import numpy as np
import cv2
from tensorflow.keras.models import load_model

EMPTY = True
NOT_EMPTY = False

MODEL = load_model("cnn_parking_model.h5")

def empty_or_not(spot_bgr):
    img_resized = cv2.resize(spot_bgr, (128, 128))
    img_resized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_resized, axis=0)
    prediction = MODEL.predict(img_input, verbose=0)[0][0]
    return EMPTY if prediction > 0.5 else NOT_EMPTY

def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1

    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        if w <= 0 or h <= 0:
            continue

        slots.append([x1, y1, w, h])

    return slots

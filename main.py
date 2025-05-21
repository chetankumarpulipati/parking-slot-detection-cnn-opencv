import cv2
import numpy as np
import os
import csv
from tensorflow.keras.models import load_model
from util import get_parking_spots_bboxes

MODEL_PATH = 'cnn_parking_model.h5'

model = load_model(MODEL_PATH)

def predict_spot_cnn(spot_img):
    img_resized = cv2.resize(spot_img, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    pred = model.predict(img_batch, verbose=0)[0][0]
    return pred > 0.5

def process_frame_with_cnn(frame, spots, spots_status):
    height, width = frame.shape[:2]

    for idx, (x, y, w, h) in enumerate(spots):
        x_end = min(x + w, width)
        y_end = min(y + h, height)

        if x >= width or y >= height or x >= x_end or y >= y_end:
            spots_status[idx] = False
            continue

        crop = frame[y:y_end, x:x_end]
        if crop.size == 0:
            spots_status[idx] = False
            continue

        status = predict_spot_cnn(crop)
        spots_status[idx] = status

        color = (0, 255, 0) if status else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    window_name = 'Parking Status'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    desired_width = 960
    desired_height = 540
    cv2.resizeWindow(window_name, desired_width, desired_height)
    cv2.imshow(window_name, frame)

def main(mask_path, input_path):
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"Failed to read mask: {mask_path}")
        return

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    spots_status = [False for _ in spots]

    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video: {input_path}")
            return

        frame_nmr = 0
        step = 30
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_nmr % step == 0:
                process_frame_with_cnn(frame, spots, spots_status)

            cv2.namedWindow('Parking Status', cv2.WINDOW_NORMAL)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame_nmr += 1

        cap.release()
        cv2.destroyAllWindows()

    else:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read image: {input_path}")
            return

        process_frame_with_cnn(image, spots, spots_status)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # After processing, prepare CSV data
    total_slots = len(spots)
    available_slots = sum(spots_status)
    occupied_slots = total_slots - available_slots

    csv_file = 'output/parking_status.csv'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Total Slots', 'Occupied Slots', 'Available Slots'])
        writer.writerow([total_slots, occupied_slots, available_slots])

    print(f"Parking status saved to {csv_file}")

if __name__ == "__main__":
    mask_path = 'mask_1920_1080.png'
    input_path = 'input.jpg'
    main(mask_path, input_path)

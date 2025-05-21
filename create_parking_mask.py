import cv2
import numpy as np

image_path = 'input.jpg'
output_txt = 'parking_spots.txt'
output_mask = 'mask_1920_1080.png'
RECT_WIDTH = 70   
RECT_HEIGHT = 30
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

spots = []

original_image = cv2.imread(image_path)
if original_image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

original_image = cv2.resize(original_image, (TARGET_WIDTH, TARGET_HEIGHT))
image = original_image.copy()


# ==== MOUSE CALLBACK ====
def mouse_click(event, x, y, flags, param):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        w, h = RECT_WIDTH, RECT_HEIGHT
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        spots.append((x1, y1, w, h))
        print(f"Spot {len(spots)}: ({x1}, {y1}, {w}, {h})")


# ==== SETUP WINDOW ====
cv2.namedWindow('Select Parking Spots', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select Parking Spots', 1280, 720)
cv2.setMouseCallback('Select Parking Spots', mouse_click)

while True:
    cv2.imshow('Select Parking Spots', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# ==== SAVE SPOTS ====
with open(output_txt, 'w') as f:
    for spot in spots:
        f.write(','.join(map(str, spot)) + '\n')
print(f"\nSaved {len(spots)} parking spots to {output_txt}")

# ==== GENERATE MASK ====
mask = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)

for x, y, w, h in spots:
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

cv2.imwrite(output_mask, mask)
print(f"Mask saved to {output_mask} with size {mask.shape}")

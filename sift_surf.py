"""SIFT (Scale-Invariant Feature Transform) y SURF (Speeded-Up Robust Features)"""
import argparse
args = argparse.ArgumentParser(
    description="Extract SIFT keypoints from an RGB image and visualize them.",
    epilog="Example usage:\n  python extract_sift_surf.py --rgb path/to/rgb.png",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
args.add_argument("--rgb", type=str, required=True, help="Path to the RGB image.")
args.add_argument("--save_dir", type=str, default="./sift_keypoints", help="Directory to save the keypoints images.")
args = args.parse_args()

import cv2
import numpy as np
import os
from datetime import datetime

rgb_path = args.rgb
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

img = cv2.imread(rgb_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
win = "SIFT Keypoints"

def update(val=None):
    num_features = cv2.getTrackbarPos("Num Features", win)
    detector = cv2.SIFT_create(nfeatures=num_features)
    keypoints, _ = detector.detectAndCompute(gray, None)
    img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(win, img_out)

cv2.namedWindow(win)
cv2.createTrackbar("Num Features", win, 300, 1000, update)

update()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        num_features = cv2.getTrackbarPos("Num Features", win)
        detector = cv2.SIFT_create(nfeatures=num_features)
        keypoints, _ = detector.detectAndCompute(gray, None)
        img_out = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"sift_keypoints_{timestamp}.png")
        cv2.imwrite(save_path, img_out)
        print(f"[INFO] Saved keypoints image to: {save_path}")

cv2.destroyAllWindows()


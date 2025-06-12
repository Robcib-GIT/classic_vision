import argparse
args = argparse.ArgumentParser(
    description="Extract surface normals from depth image and project onto RGB image.",
    epilog="Example usage:\n  python extract_normals_angles_resize_640x640.py --depth path/to/depth.png --rgb path/to/rgb.png",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
args.add_argument("--depth", type=str, required=True, help="Path to the depth image.")
args.add_argument("--rgb", type=str, required=True, help="Path to the RGB image.")
args = args.parse_args()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# === FILE PATHS ===
depth_path = args.depth
rgb_path = args.rgb

start_time = time.time()

# === D435 CAMERA INTRINSICS ===
# These are intrinsic calibration parameters for each camera (depth and color)
depth_intrinsics = {
    'fx': 421.276,
    'fy': 421.276,
    'cx': 424.0,
    'cy': 240.0
}
color_intrinsics = {
    'fx': 615.899,
    'fy': 615.899,
    'cx': 320.0,
    'cy': 240.0
}

# === EXTRINSICS FROM DEPTH TO COLOR ===
R = np.eye(3)  # Assume rotation is identity (no rotation)
T = np.array([[0.015], [0], [0]])  # Small translation between sensors (in meters)

# === FUNCTIONS ===
def depth_to_pointcloud(depth_img, intr):
    """Convert a depth image into a point cloud using depth camera intrinsics."""
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))  # Pixel grid
    z = depth_img.astype(np.float32) / 1000.0  # Convert depth to meters
    x = (i - intr['cx']) * z / intr['fx']
    y = (j - intr['cy']) * z / intr['fy']
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # Flattened array of 3D points
    return points, i.flatten(), j.flatten()

def project_to_color(points, color_intr, R, T):
    """Project 3D points into color image using projection matrix and intrinsics."""
    points_trans = (R @ points.T + T).T  # Apply extrinsics (rotation and translation)
    x, y, z = points_trans[:, 0], points_trans[:, 1], points_trans[:, 2]
    u = (x * color_intr['fx'] / z + color_intr['cx']).astype(np.int32)
    v = (y * color_intr['fy'] / z + color_intr['cy']).astype(np.int32)
    return u, v, z

# === LOAD IMAGES ===
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
rgb_img = cv2.imread(rgb_path)

if depth_img is None or rgb_img is None:
    print("Error loading images.")
    exit()

# === ALIGN DEPTH TO RGB ===
# Step 1: Convert depth to 3D points using depth intrinsics
points, i_coords, j_coords = depth_to_pointcloud(depth_img, depth_intrinsics)

# Step 2: Project those 3D points into the RGB image
u, v, z = project_to_color(points, color_intrinsics, R, T)

# Step 3: Filter out points that project outside of RGB image boundaries or behind the camera
rgb_h, rgb_w = rgb_img.shape[:2]
valid_proj = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h) & (z > 0)

# Step 4: Find the (i, j) coordinates in depth that map to valid (u, v) in RGB
depth_x = i_coords[valid_proj]
depth_y = j_coords[valid_proj]

# Step 5: Crop the depth image around the valid projection region
x_min = np.min(depth_x)
x_max = np.max(depth_x)
y_min = np.min(depth_y)
y_max = np.max(depth_y)
depth_crop = depth_img[y_min:y_max+1, x_min:x_max+1]

# Step 6: Resize the cropped depth image to the RGB resolution
depth_crop_resized = cv2.resize(depth_crop, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)

# === RESIZE BOTH IMAGES TO 640x640 ===
target_size = (640, 640)
rgb_img_resized = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_LINEAR)
depth_resized = cv2.resize(depth_crop_resized, target_size, interpolation=cv2.INTER_LINEAR)

# === NORMAL COMPUTATION ===
# Step 1: Convert depth to meters and generate mask for valid pixels
depth_m = depth_resized.astype(np.float32) / 1000.0
valid_mask = depth_m > 0

# Step 2: Reconstruct the 3D point cloud from resized depth using color intrinsics
h, w = target_size
u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
Z = depth_m
X = (u_grid - color_intrinsics['cx']) * Z / color_intrinsics['fx']
Y = (v_grid - color_intrinsics['cy']) * Z / color_intrinsics['fy']
points = np.full((h, w, 3), np.nan, dtype=np.float32)
points[valid_mask] = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=-1)

# Step 3: Compute gradients using Sobel (approx. partial derivatives along x and y)
dzdx = cv2.Sobel(points, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x (columns)
dzdy = cv2.Sobel(points, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y (rows)

# Step 4: Compute normal vectors using cross product of gradients
normals = np.cross(dzdy, dzdx)
norm = np.linalg.norm(normals, axis=2, keepdims=True)
normals = normals / (norm + 1e-8)  # Normalize to unit vectors
normals[~valid_mask] = np.nan  # Invalidate normals where depth is missing

print("Done - time >>",time.time() - start_time)

# === VISUALIZATION OF NORMALS OVER RGB ===
# We draw normals as arrows over the RGB image for visual interpretation
rgb_vis = rgb_img_resized.copy()
step = 9       # Sampling step (pixels between arrows)
length = 9     # Length scale of the arrows

# For each sampled pixel location...
for y in range(0, h, step):
    for x in range(0, w, step):
        n = normals[y, x]
        if not np.any(np.isnan(n)):  # Skip if the normal is invalid
            dx = int(n[0] * length)
            dy = int(n[1] * length)  # Invert y for image coordinates
            pt1 = (x, y)              # Start of arrow
            pt2 = (x + dx, y + dy)    # End of arrow
            cv2.arrowedLine(rgb_vis, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

# === CONVERT NORMALS (x,y,z) TO AZIMUTH AND ELEVATION ANGLES ===
azimuths = np.full((h, w), np.nan, dtype=np.float32)
elevations = np.full((h, w), np.nan, dtype=np.float32)

valid_normals = valid_mask & (~np.isnan(normals[..., 0]))

# azimuth = arctan2(nx, nz) in degrees
azimuths[valid_normals] = np.degrees(np.arctan2(normals[valid_normals, 0], normals[valid_normals, 2]))
# elevation = arcsin(ny) in degrees
elevations[valid_normals] = np.degrees(np.arcsin(normals[valid_normals, 1]))

# Stack into a 3-channel array (azimuth, elevation, 0)
angle_map = np.zeros((h, w, 2), dtype=np.float32)
angle_map[..., 0] = azimuths
angle_map[..., 1] = elevations

print(angle_map)

# === DISPLAY FINAL IMAGE ===
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB))
plt.title("Surface normals projected onto RGB image (640x640)")
plt.axis("off")
plt.tight_layout()
plt.show()




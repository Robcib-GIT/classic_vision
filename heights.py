import argparse
args = argparse.ArgumentParser(
    description="Script para extraer alturas estandarizadas de imágenes de profundidad y RGB.",
    epilog="Ejemplo de uso:\n"
           "python standarization_heights.py --depth depth.png --rgb rgb.png --voxel_size 0.01 --scale 0.5",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
args.add_argument('--depth', type=str, required=True, help='Ruta de la imagen de profundidad.')
args.add_argument('--rgb', type=str, required=True, help='Ruta de la imagen RGB.')
args.add_argument('--save_dir', type=str, default='./height', help='Directorio para guardar los resultados.')
args.add_argument('--voxel_size', type=float, default=0.01, help='Tamaño del voxel para downsampling de la nube de puntos.')
args.add_argument('--scale', type=float, default=0.5, help='Escala para redimensionar las imágenes.')
args = args.parse_args()

import cv2
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import csv

# Parámetros
VOXEL_SIZE = args.voxel_size
scale = args.scale

# RUTA DE ENTRADA DE LAS IMAGENES
depth_path = args.depth
rgb_path = args.rgb
save_dir = args.save_dir


# ************************************************************************************************************
# ************************************************************************************************************
# ******************* AJUSTE DE LA IMAGEN DE PROFUNDIDAD
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

R = np.eye(3)
T = np.array([[0.015], [0], [0]])

# === FUNCIONES ===
def depth_to_pointcloud(depth_img, intr):
    h, w = depth_img.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img.astype(np.float32) / 1000.0
    x = (i - intr['cx']) * z / intr['fx']
    y = (j - intr['cy']) * z / intr['fy']
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points, i.flatten(), j.flatten()

def project_to_color(points, color_intr, R, T):
    points_trans = (R @ points.T + T).T
    x, y, z = points_trans[:, 0], points_trans[:, 1], points_trans[:, 2]
    u = (x * color_intr['fx'] / z + color_intr['cx']).astype(np.int32)
    v = (y * color_intr['fy'] / z + color_intr['cy']).astype(np.int32)
    return u, v, z

# === CARGA DE IMÁGENES ===
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
rgb_img = cv2.imread(rgb_path)

if depth_img is None or rgb_img is None:
    print("Error cargando imágenes.")
    exit()

# Paso 1: obtener puntos 3D desde depth
points, i_coords, j_coords = depth_to_pointcloud(depth_img, depth_intrinsics)

# Paso 2: proyectarlos a imagen RGB
u, v, z = project_to_color(points, color_intrinsics, R, T)

# Paso 3: filtrar solo los píxeles proyectados dentro de la RGB
rgb_h, rgb_w = rgb_img.shape[:2]
mask = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h) & (z > 0)

# Paso 4: obtener las coordenadas (x,y) en la depth original que corresponden a RGB
depth_x = i_coords[mask]
depth_y = j_coords[mask]

# Crear máscara de área útil en la imagen depth original
mask_map = np.zeros_like(depth_img, dtype=np.uint8)
mask_map[depth_y, depth_x] = 255

# Paso 5: calcular bounding box de esa región
x_min = np.min(depth_x)
x_max = np.max(depth_x)
y_min = np.min(depth_y)
y_max = np.max(depth_y)

# Recortar directamente desde la imagen depth original
depth_crop = depth_img[y_min:y_max+1, x_min:x_max+1]

# Redimensionar recorte para que coincida con RGB si es necesario
depth_crop_resized = cv2.resize(depth_crop, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
# ******************* AJUSTE DE LA IMAGEN DE PROFUNDIDAD
# ************************************************************************************************************
# ************************************************************************************************************

# Cargar imágenes
#depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
depth_image = depth_crop_resized
rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)


if depth_image is None or rgb_image is None:
    raise FileNotFoundError("No se pudieron cargar las imágenes.")

# Redimensionar imágenes (mantener relación proporcional)
depth_resized = cv2.resize(depth_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
rgb_resized = cv2.resize(rgb_image, (depth_resized.shape[1], depth_resized.shape[0]), interpolation=cv2.INTER_AREA)

# Parámetros intrínsecos ajustados
fx = 616.0 * scale
fy = 616.0 * scale
cx = 320.0 * scale
cy = 240.0 * scale

# Generar nube de puntos sin filtrar
height, width = depth_resized.shape
u = np.tile(np.arange(width), (height, 1))
v = np.tile(np.arange(height).reshape(-1, 1), (1, width))

z = depth_resized.astype(np.float32) / 1000.0
x = (u - cx) * z / fx
y = (v - cy) * z / fy

valid = z > 0
x = x[valid]
y = y[valid]
z = z[valid]
points = np.stack((x, y, z), axis=-1)

# ======== Generar normal_map ANTES de filtrar ========
pcd_temp = o3d.geometry.PointCloud()
pcd_temp.points = o3d.utility.Vector3dVector(points)
pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd_temp.orient_normals_consistent_tangent_plane(30)

# Crear imagen de normales como RGB
normal_map = np.zeros((height, width, 3), dtype=np.uint8)
normals_np = np.asarray(pcd_temp.normals)
normal_colors = ((normals_np + 1.0) / 2.0) * 255.0
normal_colors = normal_colors.astype(np.uint8)
normal_map[valid] = normal_colors

# Almacenamiento de las normales como vectores unitarios para cada pixel
normal_unitary_map = np.zeros((height, width, 3), dtype=np.float32)
normal_unitary_array = normals_np.astype(np.float32)
normal_unitary_map[valid] = normal_unitary_array


print("normals_np:", normals_np.shape)
print("Normals color:", normal_colors.shape)
#print("Normals color:", normal_colors)
print("Forma rgb original:", rgb_image.shape)
#print("Forma rgb:", rgb_image)
print("Normals map:", normal_map.shape)
#print("Normals map:", normal_map)
print("normal_unitary_array:", normal_unitary_map.shape)


# ************************************************************************************************* Al macenar datos para analizar
# Asumimos que normal_map tiene forma (H, W, 3) y valid es (H, W)
"""H, W, _ = normal_unitary_map.shape

# Crear una matriz de strings (cada celda: "nx,ny,nz" o "nan,nan,nan")
csv_data = []
for y in range(H):
    row = []
    for x in range(W):
        if valid[y, x]:
            nx, ny, nz = normal_unitary_map[y, x]
            row.append(f"{nx:.6f},{ny:.6f},{nz:.6f}")
        else:
            row.append("nan,nan,nan")
    csv_data.append(row)

# Guardar en CSV
with open("normal_map_per_pixel_unitary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)"""
# ************************************************************************************************* Al macenar datos para analizar


# ======== Continuar con nube de puntos para visualización 3D ========
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Filtrado y downsample
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

# Estimar y orientar normales para visualización
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(30)

# ESCALA DE LAS VISUALIZACIONES
normals_vis = np.asarray(pcd.normals) * 0.1
pcd.normals = o3d.utility.Vector3dVector(normals_vis)

# Mostrar nube de puntos con normales pequeñas
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# ************************************************************************************************ Visualizacion pointcloud Open3D
o3d.visualization.draw_geometries([pcd, axes],
                                  window_name="Nube de puntos con normales pequeñas",
                                  point_show_normal=True)

# ======== Alineación y Superposición normal_map sobre la imagen RGB ========

# Asegúrate de que ambas imágenes estén del mismo tipo (uint8) y tamaño
assert rgb_resized.shape == normal_map.shape, "Las imágenes deben tener el mismo tamaño."

# Superponer normal_map sobre la imagen RGB
alpha = 0.6  # peso de la imagen RGB
beta = 0.4   # peso del normal_map
overlay = cv2.addWeighted(rgb_resized, alpha, normal_map, beta, 0)

# *********************************************************************************************** estandarizacion de alturas
# === 1. Detectar plano más cercano usando RANSAC ===
print("Inicializando estandarizacion de alturas")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plano encontrado: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

# === 2. Calcular altura relativa de cada punto al plano (distancia punto-plano) ===
# Ecuación de distancia: d = |a*x + b*y + c*z + d| / sqrt(a² + b² + c²)
numerador = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
denominador = np.sqrt(a**2 + b**2 + c**2 + d**2)
heights = numerador / denominador  # altura relativa al plano

# === 3. Crear mapa 2D de alturas ===
height_map = np.zeros((height, width), dtype=np.float32)
height_map[valid] = heights
print("mapa de alturas",height_map)
print("mapa de alturas",height_map.shape)

# (Opcional) Normalizar para visualización
norm_height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color_height_map = cv2.applyColorMap(norm_height_map, cv2.COLORMAP_JET)

# Mostrar o guardar
cv2.imshow("Altura relativa al plano base", color_height_map)
cv2.waitKey(0)


# ALMACENAMIENTO DE VALORES DE SALIDA DE LAS ALTURAS EN FORMATO CSV
"""H, W = height_map.shape
csv_data = []

for y in range(H):
    row = []
    for x in range(W):
        if valid[y, x]:
            row.append(f"{height_map[y, x]:.6f}")
        else:
            row.append("nan")  # píxeles sin información válida
    csv_data.append(row)

# Escribir en archivo CSV
with open("height_map_per_pixel.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)"""

# *********************************************************************************************************************
# *********************************************************************************************************************

# ************** REPRESENTACION DE CAMPO DE VECTORES

# *********************************************************************************************************************
# *********************************************************************************************************************


# Usaremos la imagen resized para esto
img_display = rgb_resized.copy()

# Opcional: reducir número de flechas para visualización
step = 16  # cada N pixeles
h, w = depth_resized.shape

# Reconvertir máscara 'valid' a forma 2D
valid_mask = valid.reshape(h, w)

for v in range(0, h, step):
    for u in range(0, w, step):
        if not valid_mask[v, u]:
            continue

        idx = np.sum(valid_mask[:v, :]) + np.sum(valid_mask[v, :u])
        if idx >= normals_np.shape[0]:
            continue

        # Centro de la flecha
        pt1 = (u, v)

        # Componente de la normal
        n = -normals_np[idx]  # (nx, ny, nz)

        # Proyección 2D (ignorar z, o usar solo x/y como dirección)
        scale = 13
        pt2 = (int(u + n[0]*scale), int(v + n[1]*scale))

        # Dibujar flecha sobre la imagen
        cv2.arrowedLine(img_display, pt1, pt2, color=(0, 255, 0), thickness=1, tipLength=0.3)

# *******************************************************************************************************
# ************************************************************************************** MOSTRAR IMAGENES
# *******************************************************************************************************

if len(depth_img.shape) == 2:
    depth_img_t = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_img_t = np.uint8(depth_img_t)
    depth_img_t = cv2.cvtColor(depth_img_t, cv2.COLOR_GRAY2BGR)

# Redimensionar si no tienen el mismo tamaño
if depth_img_t.shape != rgb_img.shape:
    rgb_img_t = cv2.resize(rgb_img, (depth_img_t.shape[1], depth_img_t.shape[0]))

combined = np.hstack((rgb_img_t, depth_img_t))

# Mostrar
cv2.imshow("RGB y Depth", combined)
cv2.waitKey(0)

cv2.imshow("OVERLAY", overlay)
cv2.waitKey(0)

# Mostrar resultado
cv2.imshow("Campo de normales proyectado 2D", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *******************************************************************************************************
# ************************************************************************************** MOSTRAR IMAGENES
# *******************************************************************************************************

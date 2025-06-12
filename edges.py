import argparse
parser = argparse.ArgumentParser(
    description="Extract edges from an RGB image using Canny edge detection.",
    epilog="Example: python extract_edges.py --rgb input.jpg --save_dir ./edges"
)
parser.add_argument('--rgb', type=str, required=True, help='Path to the RGB image.')
parser.add_argument('--save_dir', type=str, default='./edges', help='Directory to save the edge images.')
args = parser.parse_args()

import cv2
import os
from datetime import datetime

# Ruta de la imagen RGB
rgb_path = args.rgb

# Ruta para guardar las imágenes de bordes
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

# Cargar y convertir imagen a escala de grises
rgb = cv2.imread(rgb_path)
gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# Inicializar ventana
cv2.namedWindow('Canny Edge Detector')

# Función que actualiza el resultado de Canny
def update_canny(val):
    t1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edge Detector')
    t2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edge Detector')
    edges = cv2.Canny(gray, t1, t2)
    cv2.imshow('Canny Edge Detector', edges)

# Crear sliders
cv2.createTrackbar('Threshold 1', 'Canny Edge Detector', 50, 500, update_canny)
cv2.createTrackbar('Threshold 2', 'Canny Edge Detector', 150, 500, update_canny)

# Mostrar la imagen inicial
update_canny(None)

# Esperar interacción
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC para salir
        break

    elif key == ord('s'):  # 's' para guardar imagen
        t1 = cv2.getTrackbarPos('Threshold 1', 'Canny Edge Detector')
        t2 = cv2.getTrackbarPos('Threshold 2', 'Canny Edge Detector')
        edges = cv2.Canny(gray, t1, t2)

        # Generar nombre con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edges_{timestamp}.png"
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, edges)
        print(f"[INFO] Saved edge image to: {save_path}")

cv2.destroyAllWindows()

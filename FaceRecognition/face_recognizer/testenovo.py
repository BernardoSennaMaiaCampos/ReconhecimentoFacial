import face_recognition
import cv2
import numpy as np

img_path = "images/bernardo_teste.jpg"

img_bgr = cv2.imread(img_path)
if img_bgr is None:
    print("❌ Erro ao carregar imagem com OpenCV")
    exit()

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_rgb = cv2.resize(img_rgb, (640, 640))
image_np = np.array(img_rgb, dtype=np.uint8)
image_np = np.ascontiguousarray(image_np)

print("📐 Shape:", image_np.shape)
print("🧬 Tipo:", image_np.dtype)
print("🎨 Canais:", image_np.shape[2] if len(image_np.shape) == 3 else "N/A")
print("📌 Contíguo?", image_np.flags['C_CONTIGUOUS'])

try:
    locs = face_recognition.face_locations(image_np)
    print(f"✅ Rostos detectados: {len(locs)}")
except Exception as e:
    print("❌ Erro ao detectar rostos:", e)

from PIL import Image
import numpy as np


caminho = "images/bernardo_teste.jpg"


img = Image.open(caminho).convert("RGB")
arr = np.array(img)

print("Shape:", arr.shape)
print("Tipo:", arr.dtype)
print("Canais:", arr.shape[2] if len(arr.shape) == 3 else "Imagem sem canais")

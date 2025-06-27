import face_recognition
import cv2
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np


IMAGES_DIR = Path("images")
ENCODINGS_PATH = Path("output/encodings.pkl")
MODEL = "hog"  

def encode_known_faces():
    known_encodings = []
    known_names = []

    for file in IMAGES_DIR.glob("*.*"):
        if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        name = file.stem.split("_")[0]  
        print(f"\n🔍 Processando: {file.name} (Pessoa: {name})")

        try:
            
            img = Image.open(file).convert("RGB")
            image = np.array(img)

            
            if image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3:
                print(f"❌ Imagem inválida para reconhecimento: {file.name}")
                print(f" - dtype: {image.dtype}, shape: {image.shape}")
                continue

            face_locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, face_locations)

            if not encodings:
                print(f"❌ Nenhum rosto encontrado em {file.name}")
                continue

            known_encodings.extend(encodings)
            known_names.extend([name] * len(encodings))
            print(f"✅ {len(encodings)} rosto(s) codificado(s)")

        except Exception as e:
            print(f"❌ Erro ao processar {file.name}: {e}")

    
    ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"\n✅ Codificação concluída. Total de rostos salvos: {len(known_encodings)}")

if __name__ == "__main__":
    encode_known_faces()

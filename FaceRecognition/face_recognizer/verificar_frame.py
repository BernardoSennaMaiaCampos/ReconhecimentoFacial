import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ret, frame = cap.read()
if not ret or frame is None:
    print("Erro ao capturar frame.")
else:
    print("Shape:", frame.shape)
    print("Tipo:", frame.dtype)
    print("Canais:", frame.shape[2] if len(frame.shape) > 2 else "Monocromático")

cap.release()

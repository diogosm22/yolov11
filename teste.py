import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv11n
model = YOLO('C:/Users/diogo/Desktop/python/yolov11/best.pt')

# Iniciar a captura de vídeo da câmera (0 é geralmente a câmera padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Fazer a predição no frame capturado com nível de confiança ajustado
    results = model(frame, conf=0.8)  # Ajusta a confiança para 50%

    # Desenhar as caixas delimitadoras e rótulos no frame
    annotated_frame = results[0].plot()

    # Exibir o frame anotado
    cv2.imshow('Detecção de Objetos', annotated_frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()

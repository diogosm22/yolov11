import time

import cv2
import ncnn
import numpy as np
import torch
from picamera2 import Picamera2


def test_inference():
    # Cria uma entrada de exemplo (imagem capturada pela câmera)
    in0 = np.random.rand(1, 3, 640, 640).astype(np.float32)

    out = []

    # Inicializa o modelo ncnn
    net = ncnn.Net()
    # Carrega os arquivos param e bin
    net.load_param("C:/path/to/model.ncnn.param")
    net.load_model("C:/path/to/model.ncnn.bin")

    # Cria o extrator e realiza a inferência
    ex = net.create_extractor()
    # Transforma a entrada (imagem) para o formato ncnn
    ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

    # Extrai a saída
    ret, out0 = ex.extract("out0")
    out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    # Retorna o resultado
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

def capture_image():
    # Inicializa a câmera PiCamera2
    picam2 = Picamera2()
    picam2.start()

    # Aguardar estabilização da câmera
    time.sleep(2)

    # Captura a imagem
    image = picam2.capture_array()

    # Redimensionar para o formato de entrada do modelo (640x640)
    image_resized = cv2.resize(image, (640, 640))

    # Normaliza a imagem e converte para um formato que o ncnn aceite
    in0 = image_resized / 255.0
    in0 = np.transpose(in0, (2, 0, 1))  # Converte de HWC para CHW
    return in0.astype(np.float32)

if __name__ == "__main__":
    # Captura uma imagem da câmera
    in0 = capture_image()
    
    # Realiza a inferência com o modelo ncnn
    output = test_inference()

    print("Output:", output)

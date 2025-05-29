# Librerías necesarias
import numpy as np
import torch
from PIL import Image
import mmcv

# Funciones y clases de mmsegmentation
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules
from mmengine.model import revert_sync_batchnorm

# ------------------------------------------------------------------------
# 🧠 Carga el modelo de segmentación a partir del archivo de configuración
# y los pesos preentrenados. Registra los módulos necesarios y ajusta
# la normalización por lotes si se trabaja en CPU.
# ------------------------------------------------------------------------
def load_model(config_path: str, checkpoint_path: str, device: str = "cpu"):
    register_all_modules()  # Registra backbones, heads, etc.
    model = init_model(config_path, checkpoint_path, device=device)  # Carga el modelo

    if device == "cpu":
        model = revert_sync_batchnorm(model)  # Ajusta capas BatchNorm para CPU (SyncBN no soportado)

    return model

# ------------------------------------------------------------------------
# 🧠 Preprocesamiento de la imagen:
# Convierte la imagen PIL a array NumPy, formato requerido por mmseg.
# ------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    return np.array(image)

# ------------------------------------------------------------------------
# 🧠 Infiere el mapa de segmentación sobre la imagen de entrada.
# Utiliza el modelo cargado previamente.
# ------------------------------------------------------------------------
def run_inference(model, input_image: np.ndarray):
    return inference_model(model, input_image)

# ------------------------------------------------------------------------
# 🧠 Postprocesa la salida del modelo:
# Convierte el mapa de clases a una imagen RGB utilizando la paleta del modelo
# y superpone el resultado sobre la imagen original.
# ------------------------------------------------------------------------
def postprocess_output(result, original_image: Image.Image, model) -> Image.Image:
    # Extrae el mapa de predicción (clases por píxel)
    overlay = result.pred_sem_seg.data.cpu().numpy().squeeze()

    # Obtiene la paleta de colores desde el modelo
    palette = model.dataset_meta['palette']  # lista de [R, G, B] por clase

    # Crea imagen RGB vacía del mismo tamaño que overlay
    seg_map = np.zeros((overlay.shape[0], overlay.shape[1], 3), dtype=np.uint8)

    # Pinta cada píxel según su clase con el color correspondiente
    for label, color in enumerate(palette):
        seg_map[overlay == label] = color

    # Redimensiona la máscara segmentada al tamaño de la imagen original
    seg_map_resized = mmcv.imresize(seg_map, (original_image.width, original_image.height))

    # Convierte a imagen PIL y combina con la original con transparencia
    seg_image = Image.fromarray(seg_map_resized).convert("RGB")
    return Image.blend(original_image.convert("RGB"), seg_image, alpha=0.6)

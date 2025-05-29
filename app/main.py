# Librerías principales
import streamlit as st
from PIL import Image
import numpy as np

# Funciones utilitarias propias del proyecto
from app.utils import load_model, preprocess_image, run_inference, postprocess_output

# Configura los parámetros de la interfaz de Streamlit
st.set_page_config(page_title="Segmentación con PVT", layout="wide")
st.title("Segmentación de Imágenes con Pyramid Vision Transformer")

# ------------------------------------------------------------------------
# Carga del modelo una sola vez para evitar múltiples inicializaciones
# Esto es útil para eficiencia al trabajar con Streamlit.
# Se cachea el modelo utilizando el decorador @st.cache_resource
# ------------------------------------------------------------------------
@st.cache_resource
def load():
    return load_model(
        config_path="model/configs/pvt_config.py",     # Ruta al archivo .py con arquitectura del modelo
        checkpoint_path="model/weights/pvtv2.pth",     # Pesos preentrenados de PVT v2
        device="cpu"                                   # Usar "cuda:0" si se tiene GPU disponible
    )

# Carga el modelo al iniciar la aplicación
model = load()

# ------------------------------------------------------------------------
# Panel lateral de Streamlit para que el usuario cargue una imagen
# ------------------------------------------------------------------------
st.sidebar.header("Cargar Imagen")
uploaded_file = st.sidebar.file_uploader(
    "Selecciona una imagen", type=["jpg", "jpeg", "png"]
)

# ------------------------------------------------------------------------
# Si el usuario carga una imagen, se ejecuta todo el pipeline:
# visualización, preprocesamiento, inferencia y postprocesamiento
# ------------------------------------------------------------------------
if uploaded_file:
    # Carga y convierte la imagen a RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Muestra la imagen original
    st.image(image, caption="Imagen Original", use_container_width=True)

    # Preprocesa la imagen para adaptarla al modelo (array numpy)
    input_tensor = preprocess_image(image)

    # Realiza inferencia usando el modelo cargado
    output = run_inference(model, input_tensor)

    # Postprocesa el resultado y lo convierte a imagen superpuesta
    result_img = postprocess_output(output, image, model)

    # Muestra la imagen segmentada al usuario
    st.subheader("Segmentación Generada")
    st.image(result_img, use_container_width=True)

# Si no se ha cargado imagen, se muestra mensaje informativo
else:
    st.info("Por favor sube una imagen para segmentar.")

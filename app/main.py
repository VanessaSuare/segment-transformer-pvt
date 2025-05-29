import streamlit as st
from PIL import Image
import numpy as np
from app.utils import load_model, preprocess_image, run_inference, postprocess_output

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configura la página de Streamlit
st.set_page_config(page_title="Segmentación con PVT", layout="wide")
st.title("Segmentación de Imágenes con Pyramid Vision Transformer")

# Carga el modelo preentrenado una sola vez para eficiencia
@st.cache_resource
def load():
    return load_model(
        config_path="model/configs/pvt_config.py",
        checkpoint_path="model/weights/pvtv2.pth",
        device="cpu"  # Cambiar a 'cuda:0' si se usa GPU
    )

model = load()

# 🧠 Subida de imagen desde la interfaz Streamlit
st.sidebar.header("Cargar Imagen")
uploaded_file = st.sidebar.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

# 📷 Segmentación en vivo desde cámara
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model(
            config_path="model/configs/pvt_config.py",
            checkpoint_path="model/weights/pvtv2.pth",
            device="cpu"
        )

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        img_pil = Image.fromarray(image[..., ::-1])
        input_tensor = preprocess_image(img_pil)
        result = run_inference(self.model, input_tensor)
        result_img = postprocess_output(result, img_pil, self.model)
        return np.array(result_img)

st.header("📷 Segmentación en Tiempo Real desde Cámara")
webrtc_streamer(key="realtime", video_transformer_factory=VideoTransformer)

# Imagen cargada manualmente
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen Original", use_container_width=True)

    input_tensor = preprocess_image(image)
    output = run_inference(model, input_tensor)
    result_img = postprocess_output(output, image, model)

    st.subheader("Segmentación Generada")
    st.image(result_img, use_container_width=True)
else:
    st.info("Por favor sube una imagen o usa la cámara para segmentar.")

# 🧠 Mostrar leyenda de clases
st.sidebar.markdown("###Leyenda de Clases")
class_names = model.dataset_meta['classes']
palette = model.dataset_meta['palette']

for cls_name, color in zip(class_names, palette):
    r, g, b = color
    color_hex = f'#{r:02x}{g:02x}{b:02x}'
    st.sidebar.markdown(
        f'<div style="display: flex; align-items: center;">'
        f'<div style="width: 20px; height: 20px; background-color:{color_hex}; margin-right:10px; border: 1px solid #000;"></div>'
        f'<span>{cls_name}</span></div>',
        unsafe_allow_html=True
    )

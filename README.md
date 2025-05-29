Aquí tienes un `README.md` completo, profesional y claro para tu proyecto de segmentación con Pyramid Vision Transformer (PVT). Este documento está listo para subir a GitHub o entregar como parte del proyecto final.

---

````markdown
# Segmentación de Imágenes con Pyramid Vision Transformer (PVT)

Este proyecto implementa un sistema de segmentación semántica de imágenes utilizando el modelo **Pyramid Vision Transformer (PVTv2)**, aprovechando la biblioteca [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). La aplicación cuenta con una interfaz interactiva desarrollada en Streamlit y está lista para ejecutarse en contenedores Docker.

## Objetivos

- Implementar y ejecutar el modelo **PVTv2** para segmentación semántica.
- Utilizar configuraciones y pesos del repositorio oficial.
- Desplegar una interfaz sencilla para usuarios finales.
- Documentar todo el flujo: carga de modelos, inferencia y visualización.

---

## Basado en

- 📄 Paper original: [Pyramid Vision Transformer](https://paperswithcode.com/paper/pyramid-vision-transformer-a-versatile)
- 💻 Repositorio base: [https://github.com/whai362/PVTv2-Seg](https://github.com/whai362/PVTv2-Seg)
- ⚙️ Framework MMSegmentation: [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

---

## 🚀 Instalación (modo Docker)

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu_usuario/segment-transformer-pvt.git
   cd segment-transformer-pvt
````

2. Construye la imagen Docker:

   ```bash
   docker build -t segment-transformer .
   ```

3. Ejecuta la aplicación:

   ```bash
   docker run -p 8501:8501 segment-transformer
   ```

4. Accede en tu navegador:

   ```
   http://localhost:8501
   ```

---

## Estructura del Proyecto

```
segment-transformer-pvt/
├── app/
│   ├── main.py              # Interfaz Streamlit
│   ├── utils.py             # Funciones de carga, inferencia y postprocesamiento
├── model/
│   ├── configs/             # Configuración del modelo (pvt_config.py)
│   ├── weights/             # Pesos preentrenados (pvtv2.pth)
├── requirements.txt         # Dependencias del proyecto
├── Dockerfile               # Imagen reproducible para ejecución
└── README.md
```

---

## Ejemplo de Uso

1. Carga una imagen desde la interfaz.
2. El modelo segmenta objetos presentes.
3. Se muestra la segmentación superpuesta en colores sobre la imagen original.

---

## ⚙️ Tecnologías Usadas

* **Python 3.10**
* **PyTorch 2.0**
* **MMSegmentation 1.0+**
* **Streamlit**
* **Docker**
* **OpenCV, Pillow, NumPy**

---

## ✍️ Autores

* Dayanna Vanessa Suarez Mazuera
* Juan Camilo Buitrago Gonzalez

---

## 📄 Licencia

@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
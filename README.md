# Segmentación de Imágenes con Pyramid Vision Transformer (PVT)

Este proyecto presenta una aplicación completa de segmentación semántica de imágenes basada en redes Transformer, específicamente el modelo **Pyramid Vision Transformer v2 (PVTv2)**. Esta arquitectura fue diseñada para resolver las limitaciones de los Vision Transformers clásicos, introduciendo mecanismos jerárquicos que permiten un manejo más eficiente de imágenes de alta resolución y estructuras espaciales complejas.

La solución se basa en el framework [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), ampliamente utilizado en tareas de visión por computadora. Además, se ha desarrollado una interfaz amigable utilizando **Streamlit**, permitiendo a cualquier usuario cargar imágenes y visualizar los resultados segmentados en tiempo real. La aplicación completa es portable y reproducible mediante **Docker**.

---

## Características destacadas

- ✔️ Modelo **PVTv2** implementado y funcional
- ✔️ Visualización en tiempo real con Streamlit
- ✔️ Inferencia sobre imágenes cargadas por el usuario
- ✔️ Contenedor Docker reproducible

---

## Basado en

- Paper: [Pyramid Vision Transformer](https://paperswithcode.com/paper/pyramid-vision-transformer-a-versatile)
- Repo base: [https://github.com/whai362/PVTv2-Seg](https://github.com/whai362/PVTv2-Seg)
- Framework: [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

---

## Instalación y ejecución (Docker)

### 1. Clona el repositorio

```bash
git clone https://github.com/VanessaSuare/segment-transformer-pvt.git
cd segment-transformer-pvt
```

### 2. Construye la imagen Docker

```bash
docker build -t segment-transformer .
```

### 3. Ejecuta la aplicación

```bash
docker run -p 8501:8501 segment-transformer
```

### 4. Abre la app en tu navegador

```
http://localhost:8501
```

---

## Estructura del Proyecto

```
segment-transformer-pvt/
├── app/
│   ├── main.py              # Interfaz con Streamlit
│   ├── utils.py             # Funciones de carga, inferencia y visualización
├── model/
│   ├── configs/             # Configuración del modelo (pvt_config.py)
│   └── weights/             # Pesos preentrenados (pvtv2.pth)
├── requirements.txt         # Dependencias del entorno
├── Dockerfile               # Docker reproducible
├── .gitignore               # Ignorar archivos pesados y temporales
└── README.md                # Este documento
```

---

## Ejemplo de uso

1. El usuario carga una imagen urbana o enciende su camara desde la interfaz.
2. El modelo la segmenta automáticamente usando PVTv2.
3. Se superpone el resultado con colores por clase.

---

## Consideraciones Técnicas

- Se usa `mmsegmentation` para cargar modelos y realizar inferencias.
- Se adaptó el archivo de configuración del modelo y se integró el checkpoint `.pth`.
- La interfaz fue desarrollada en Streamlit, incluyendo carga de imagen, procesamiento y visualización.

---

## Tecnologías

- Python 3.10
- PyTorch 2.0
- MMSegmentation
- Streamlit
- Docker
- OpenCV, Pillow, NumPy

---

## Autores

- **Dayanna Vanessa Suarez Mazuera / VanessaSuare**  
- **Juan Camilo Buitrago Gonzalez / juancbuitrago**  

---
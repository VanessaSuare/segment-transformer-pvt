FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgl1-mesa-glx libgtk2.0-dev \
    gcc g++ make wget unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instala dependencias base primero (estas casi nunca cambian)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir mmcv==2.0.1 mmengine==0.8.4 && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copia solo el código del proyecto (esto puede cambiar frecuentemente)
WORKDIR /app
COPY . /app
ENV PYTHONPATH="/app"

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py"]

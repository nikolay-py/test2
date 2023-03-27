FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Установка libgl1-mesa-glx для работы библиотеки OpenCV (cv2)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        && add-apt-repository universe \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Установка Python
RUN pip3 install --upgrade pip
RUN pip3 install wheel
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

WORKDIR /app

ARG GUNICORN_TIMEOUT
ENV GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT}

ENTRYPOINT gunicorn --workers=1 --timeout=${GUNICORN_TIMEOUT} --bind 0.0.0.0:8000 wsgi:app

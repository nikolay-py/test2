FROM nvidia/cuda:11.4.1-base-ubuntu20.04

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    python3.8-dev \
    python3.8-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Python
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && python3.8 -m pip install pip \
    && python3.8 -m pip install wheel


# Установка дополнительных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        && add-apt-repository universe \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

WORKDIR /app

ARG GUNICORN_TIMEOUT
ENV GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT}

ENTRYPOINT gunicorn --workers=1 --timeout=${GUNICORN_TIMEOUT} --bind 0.0.0.0:8000 wsgi:app

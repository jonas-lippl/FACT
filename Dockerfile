FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-dev \
    python3-pip \
    libffi-dev \
    build-essential \
    rsync

WORKDIR /mnt

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

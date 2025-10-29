FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# system deps for pillow and general tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libjpeg62-turbo-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Install CPU-only PyTorch and torchvision (adjust if you want CUDA)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

# Bind to the port provided by the environment (Railway sets $PORT)
# Use sh -c so $PORT is expanded at container runtime
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4"]


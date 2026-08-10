# Imagen base ligera con Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias primero (aprovecha cache de Docker si el codigo cambia pero no las deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el codigo del proyecto y el modelo ya entrenado
COPY src/ ./src/
COPY models/ ./models/

# El modelo se sirve desde models/lightgbm_model.txt (generado por src/train.py)
ENV MODEL_PATH=models/lightgbm_model.txt
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck nativo de Docker, util para orquestadores (docker-compose, k8s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.8-slim

WORKDIR /app

# instala unzip
RUN apt-get update && apt-get install -y unzip


# Copiar el archivo kaggle.json al directorio de configuración de Kaggle
# Asegúrate de que kaggle.json se encuentre en el mismo directorio que tu Dockerfile
COPY kaggle.json /root/.kaggle/kaggle.json

# Ajustar los permisos (requerido por kaggle)
RUN chmod 600 /root/.kaggle/kaggle.json

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script
COPY run.py .

CMD ["python", "run.py"]

FROM python:3.8-slim

WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el script de ejecución
COPY run.py .

CMD ["python", "run.py"]

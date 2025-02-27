FROM python:3.8-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py .

CMD ["python", "run.py"]

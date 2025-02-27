import os
import csv
import random

# Directorio donde se guardarán los CSV muestreados
output_dir = "/shared_volume"
os.makedirs(output_dir, exist_ok=True)

def reservoir_sample(filename, sample_size, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    reservoir = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Leer la cabecera
        for i, row in enumerate(reader):
            if i < sample_size:
                reservoir.append(row)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = row
    return header, reservoir

# Descargar el dataset con Kaggle
csv_file = 'HI-Medium_Trans.csv'
if not os.path.exists(csv_file):
    os.system("kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Medium_Trans.csv")
    os.system("unzip HI-Medium_Trans.csv.zip")

# Parámetros para el muestreo
sample_size = 10000
n_samples = 10

for i in range(n_samples):
    header, sample = reservoir_sample(csv_file, sample_size, random_state=i)
    output_filename = os.path.join(output_dir, f'sample_{i}.csv')
    with open(output_filename, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        writer.writerows(sample)
    print(f"Muestra {i} guardada en {output_filename}")

print("Data ingestion completado.")

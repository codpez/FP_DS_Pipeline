import os
import time
import glob
import pickle
import pandas as pd
from fitter import Fitter
from sklearn.preprocessing import MinMaxScaler

shared_dir = "/shared_volume"
processed_csv_folder = os.path.join(shared_dir, "processed_csv")
processed_pickle_folder = os.path.join(shared_dir, "processed_pickles")
os.makedirs(processed_csv_folder, exist_ok=True)
os.makedirs(processed_pickle_folder, exist_ok=True)

def process_new_samples():
    # Buscar archivos sample_*.csv en shared_dir que no estén en processed_csv_folder
    sample_files = glob.glob(os.path.join(shared_dir, "sample_*.csv"))
    for file in sample_files:
        # Si el archivo ya fue movido, lo saltamos
        if os.path.dirname(file) == processed_csv_folder:
            continue
        try:
            df_sample = pd.read_csv(file)
        except Exception as e:
            print(f"Error al leer {file}: {e}")
            continue
        
        # Convertir columnas relevantes a numérico
        for col in ['Amount Received', 'Amount Paid', 'To Bank', 'From Bank']:
            df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
        # Eliminar columnas innecesarias
        for col in ['Timestamp', 'Is Laundering']:
            if col in df_sample.columns:
                df_sample.drop(col, axis=1, inplace=True)
        
        # Ajuste de distribución en 'Amount Received'
        data = df_sample['Amount Received'].dropna().values
        f = Fitter(data, distributions=['norm', 'expon', 'lognorm'])
        f.fit()
        best_fit = f.get_best()
        
        # Normalización usando MinMaxScaler
        scaler = MinMaxScaler()
        float_cols = df_sample.select_dtypes(include=['float64']).columns
        scaled_values = scaler.fit_transform(df_sample[float_cols])
        df_sample[float_cols] = scaled_values
        
        # Guardar el resultado en un pickle individual
        sample_id = os.path.basename(file).replace(".csv", "")
        pickle_path = os.path.join(processed_pickle_folder, f"{sample_id}.pkl")
        with open(pickle_path, 'wb') as f_out:
            pickle.dump({
                "dataframe": df_sample,
                "distribution": best_fit
            }, f_out)
        print(f"Procesado {file}. Pickle guardado en {pickle_path}")
        
        # Mover el archivo CSV procesado a processed_csv_folder
        new_csv_path = os.path.join(processed_csv_folder, os.path.basename(file))
        os.rename(file, new_csv_path)
        print(f"Movido {file} a {new_csv_path}")

while True:
    process_new_samples()
    time.sleep(10)  # Espera 10 segundos antes de volver a buscar nuevos archivos

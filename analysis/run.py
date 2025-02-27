import os
import time
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score

shared_dir = "/shared_volume"
processed_pickle_folder = os.path.join(shared_dir, "processed_pickles")
analyzed_pickle_folder = os.path.join(shared_dir, "analyzed_pickles")
os.makedirs(analyzed_pickle_folder, exist_ok=True)

scores_csv = os.path.join(shared_dir, "silhouette_scores.csv")
if not os.path.exists(scores_csv):
    pd.DataFrame(columns=["sample_id", "ward_silhouette", "spectral_silhouette"]).to_csv(scores_csv, index=False)

def analyze_new_pickles():
    pickle_files = glob.glob(os.path.join(processed_pickle_folder, "*.pkl"))
    for pickle_file in pickle_files:
        sample_id = os.path.basename(pickle_file).replace(".pkl", "")
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error al cargar {pickle_file}: {e}")
            continue
        
        df = data["dataframe"]
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
        if df_numeric.empty:
            continue
        
        # Reducir a 2 componentes con PCA para clustering
        pca_2d = PCA(n_components=2)
        pca_result = pca_2d.fit_transform(df_numeric)
        df_pca = pd.DataFrame(pca_result, columns=['PC1','PC2'])
        print(f"Varianza explicada en 2 componentes ({sample_id}): {pca_2d.explained_variance_ratio_}")
        
        # Clustering: Agglomerative (Ward)
        ward_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
        ward_labels = ward_cluster.fit_predict(df_pca[['PC1','PC2']])
        df_pca['WardCluster'] = ward_labels
        ward_score = silhouette_score(df_pca[['PC1','PC2']], ward_labels)
        plt.figure(figsize=(8,6))
        plt.scatter(df_pca['PC1'], df_pca['PC2'], c=ward_labels, cmap='viridis', alpha=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Ward Clustering - {sample_id}')
        ward_plot_path = os.path.join(shared_dir, f"ward_{sample_id}.png")
        plt.savefig(ward_plot_path)
        plt.close()
        
        # Clustering: Spectral
        spectral_cluster = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
        spectral_labels = spectral_cluster.fit_predict(df_pca[['PC1','PC2']])
        df_pca['SpectralCluster'] = spectral_labels
        spectral_score = silhouette_score(df_pca[['PC1','PC2']], spectral_labels)
        plt.figure(figsize=(8,6))
        plt.scatter(df_pca['PC1'], df_pca['PC2'], c=spectral_labels, cmap='viridis', alpha=0.5)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Spectral Clustering - {sample_id}')
        spectral_plot_path = os.path.join(shared_dir, f"spectral_{sample_id}.png")
        plt.savefig(spectral_plot_path)
        plt.close()
        
        # Registrar resultados en el CSV
        new_entry = pd.DataFrame({
            "sample_id": [sample_id],
            "ward_silhouette": [ward_score],
            "spectral_silhouette": [spectral_score]
        })
        new_entry.to_csv(scores_csv, mode='a', header=False, index=False)
        print(f"Analizado {sample_id}: Ward: {ward_score:.2f}, Spectral: {spectral_score:.2f}")
        
        # Mover el pickle analizado a la carpeta analyzed
        new_pickle_path = os.path.join(analyzed_pickle_folder, os.path.basename(pickle_file))
        os.rename(pickle_file, new_pickle_path)

while True:
    analyze_new_pickles()
    time.sleep(10)  # Espera 10 segundos antes de buscar nuevos pickles

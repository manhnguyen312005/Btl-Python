import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
from kneed import KneeLocator

def setup_output_folder(base_path='D:/btl/results'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return base_path

def preprocess_data(dataframe):
    numeric_columns = [col for col in dataframe.columns if col not in ['Name', 'Nation', 'Squad', 'Pos']]
    data = dataframe[numeric_columns].copy()
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def compute_metrics(scaled_data, k_values=range(2, 11)):
    distortions = []
    silhouette_values = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(scaled_data)
        distortions.append(model.inertia_)
        cluster_labels = model.labels_
        silhouette_values.append(silhouette_score(scaled_data, cluster_labels))
    return distortions, silhouette_values

def plot_metrics(k_values, distortions, silhouette_values, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(k_values, distortions, 'bo-')
    plt.title('Elbow Curve')
    plt.xlabel('Clusters (k)')
    plt.ylabel('Distortion')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(k_values, silhouette_values, 'go-')
    plt.title('Silhouette Metric')
    plt.xlabel('Clusters (k)')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_plot.png'))
    plt.close()

def apply_clustering(scaled_data, optimal_clusters):
    kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    return kmeans_model.fit_predict(scaled_data)

def visualize_pca(scaled_data, cluster_labels, save_path):
    pca_model = PCA(n_components=2)
    reduced_data = pca_model.fit_transform(scaled_data)
    variance_ratio = pca_model.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.title('PCA Visualization of Clusters')
    plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}% variance)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'pca_visualization.png'))
    plt.close()
    
    return variance_ratio

def summarize_clusters(dataframe, cluster_labels, save_path):
    dataframe['ClusterID'] = cluster_labels
    summary = dataframe.groupby('ClusterID').agg({
        'Pos': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'Gls': 'mean',
        'Ast': 'mean',
        'xG': 'mean',
        'xAG': 'mean',
        'Min': 'mean',
        'Squad': lambda x: x.value_counts().index[0] if not x.value_counts().empty else 'Unknown'
    }).reset_index()
    summary.to_csv(os.path.join(save_path, 'cluster_stats.csv'), index=False)
    return summary

def run_analysis():
    output_dir = setup_output_folder()
    try:
        dataset = pd.read_csv('D:/btl/results.csv')
    except FileNotFoundError:
        print("Data file not found!")
        return
    if dataset.empty:
        print("Dataset is empty!")
        return
    processed_data = preprocess_data(dataset)
    k_range = range(2, 11)
    distortions, silhouette_values = compute_metrics(processed_data, k_range)
    plot_metrics(k_range, distortions, silhouette_values, output_dir)
    elbow_finder = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
    best_k = elbow_finder.knee
    cluster_assignments = apply_clustering(processed_data, best_k)
    pca_variance = visualize_pca(processed_data, cluster_assignments, output_dir)
    print(f"PCA Variance: {pca_variance.sum():.2f} (PC1: {pca_variance[0]:.2f}, PC2: {pca_variance[1]:.2f})")
    cluster_stats = summarize_clusters(dataset, cluster_assignments, output_dir)
    print(f"Clustered into {best_k} groups")
    print("Cluster Stats:")
    print(cluster_stats)
    dataset.to_csv(os.path.join(output_dir, 'clustered_data.csv'), index=False)
    print("Output saved in results folder")

if __name__ == "__main__":
    run_analysis()
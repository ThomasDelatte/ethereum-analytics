import itertools
import random
import time

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def data_preprocessing(df):
    """Takes a DataFrame, removes columns and applies transformations (log, scale, pca)"""
    # Strip address and label columns
    data = df.iloc[:,1:-1]
    log = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    scale = StandardScaler()
    pca = PCA(n_components=data.shape[1])
    
    pipe = Pipeline([('log', log ),
                     ('scale', scale),
                     ('PCA', pca)])
    
    processed_data = pipe.fit_transform(data)
    return processed_data

def plot_silhouette_scores(data, min_clusters, max_clusters)
    """Calculate silhouette scores for each k number of clusters and plot it."""
    
    silhouette_scores = [] 
    K = range(min_clusters, max_clusters) 

    for k in K:
        clusterer = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=0)
        preds = clusterer.fit_predict(data)
        score = silhouette_score(data, preds)
        silhouette_scores.append(score)

    # Lineplot using silhouette score
    plt.plot(K, silhouette_scores) 
    plt.title('The Silhouette Method') 
    plt.xlabel('K - Number of Clusters') 
    plt.ylabel('Silhouette score') 
    plt.show()

def make_clusters(processed_data, n_clusters):
    """Takes processed data, the desired number of clusters and returns results."""
    cl = KMeans(n_clusters=n_clusters, n_init=20, max_iter=500, n_jobs=-1, verbose=0)
    return cl.fit(processed_data)

def calc_tsne(processed_data, perplexity=20, n_components=2):
    """Dimensionality reduction by calculating t-SNE."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=500, learning_rate=100)
    tsne_results = tsne.fit_transform(processed_data)
    return tsne_results

def plot_tsne(clusters, tsne_results):
    """Plot the clusters found by reducing dimensions with calc_tsne."""
    
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)

    for cluster in np.unique(clusters):
        mask = clusters == cluster
        plt.scatter(tsne_results[mask][:,0], tsne_results[mask][:,1], s=20, alpha=.5, label=cluster)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for lh in legend.legendHandles: 
        lh.set_alpha(1)

    plt.title("Clusters: T-SNE", fontsize=20)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()
    
def plot_tsne_with_labels(tsne_results, df, dflabel, categs, colors):
    """Plot the clusters but only highlighting the data points with labels.
    Credit to Will Price for inspiration (https://github.com/willprice221/code2vec)"""
    
    labeled_addresses = dflabel["ethereum_address"].values
    labelmask = np.array([addr in labeled_addresses for addr in df["ethereum_address"] ] )
    
    # Helper function for category mask
    def cat(addr, labeled_addresses, dflabel):
        if addr not in labeled_addresses:
            return False
        else:
            idx = int(np.where(labeled_addresses==addr)[0][0])
            return dflabel.iloc[idx, 1]

    subset, not_subset  = tsne_results[labelmask] , tsne_results[~labelmask]
    fig = plt.figure(figsize=(15,12))
    #not labelled points
    plt.scatter(not_subset[:,0], not_subset[:,1], s=20, c='gray', alpha=.3)

    #categories
    cats = np.array([cat(addr, labeled_addresses, dflabel) for addr in df["ethereum_address"]]) #[address_mask] ]) #added address mask for all clusters

    for c in list(dflabel["Entity"].unique()):
        mask = dflabel["Entity"]==c

        #category mask
        catmask = cats == c

        if c in categs:
            idx=categs.index(c)
            color = colors[idx]

            plt.scatter(tsne_results[(labelmask & catmask)][:,0], tsne_results[(labelmask & catmask)][:,1], s=20, c=color, alpha=1, label=c)

    legend = plt.legend(bbox_to_anchor=(1, 1))
    for lh in legend.legendHandles: 
        lh.set_alpha(1)

    plt.title("Clusters - Labeled Data Points: T-SNE", fontsize=20)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()
    
def add_clusters_to_df(df, clusters):
    """Add the found clusters to the data."""
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = -1
    for i, row in df_with_clusters.iterrows():
        df_with_clusters.iat[i, -1] = clusters[i]
    return df_with_clusters

def show_distribution_of_clusters(clusters, dflabel, category):
    """Show how labeled data is distributed among the clusters."""
    type_cluster = 0
    num_of_type = 0
    lbl_density = 0
    print(category)
    for clust in np.unique(clusters):
        size_of_cluster = np.sum(clusters==clust)
        d = dflabel[dflabel["cluster"]==clust]
        num = np.sum(d["Entity"]==category)
        density = num / size_of_cluster * 100
        if num > num_of_type:
            lbl_density=density
            num_of_type = num
            type_cluster = clust
        print(f"Cluster number {clust} has {size_of_cluster} addresses, including {num} addresses labeled as {category} (label density: {density}).")   
    return None
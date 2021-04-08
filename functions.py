import itertools
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def data_pipeline(df):
    # Strip address and label columns
    data = df.iloc[:,1:-1]
    log = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=True)
    scale = StandardScaler()
    pca = PCA(n_components=data.shape[1])
    
    # Build pipeline
    pipe = Pipeline([('log', log ),
                     ('scale', scale ),
                     ('PCA', pca)])
    results = pipe.fit_transform(data)
    
    return pipe, results

def cluster(results, n_clusters):
    cl = KMeans(n_clusters, n_init=20, max_iter=500, n_jobs=-1, verbose=0)
    return cl.fit(results)

def calc_tsne(results, n_components=2, perplexity=20, n_iter=300):
    '''
    Calculated tsne for dataset'''
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, learning_rate=100)
    tsne_results = tsne.fit_transform(results)
    return tsne_results

def plot_tsne(clusters, tsne_results):
    '''
    plot'''
    
    cm = plt.get_cmap('nipy_spectral')

    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)

    for c in np.unique(clusters):
        mask = clusters ==c
        if np.sum(mask) <1:
            lbl = '_nolegend_'
        else:
            lbl = c
        plt.scatter(tsne_results[mask][:,0], tsne_results[mask][:,1], s=20, alpha=.4,label=lbl)

    leg = plt.legend(bbox_to_anchor=(1, 1))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.title('T-SNE', fontsize=20)
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.show()
    
    def plot_tsne_with_labels(tsne_results,df, dflabel,categs,colors):
    #need to mask df based on which results were kept from the reclustering
    
    labeled_addresses = dflabel["ethereum_address"].values
    labelmask = np.array([addr in labeled_addresses for addr in df["ethereum_address"] ] )
    
    #helper function for category mask
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
        if np.sum(mask) < 1:
            lbl = '_nolegend_'
        else:
            lbl = c

        #category mask
        catmask = cats == c

        if c in categs:
            idx=categs.index(c)
            color = colors[idx]

            plt.scatter(tsne_results[(labelmask & catmask)][:,0], tsne_results[(labelmask & catmask)][:,1], s=20, c=color, alpha=1, label=lbl)

    leg = plt.legend(bbox_to_anchor=(1, 1))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.title('T-SNE', fontsize=20)
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    plt.show()
    
def assign_cluster_to_data(df, clusters):
    df["cluster"] = 10
    for i, row in df.iterrows():
        df.iat[i, 30] = clusters[i]
    return None

def find_category_of_cluster(clusters, dflabel, category="Exchange"):
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
        print(f"Cluster number   {clust}   number of type found: {num}    cluster size: {size_of_cluster}   label density: {density}")
    return type_cluster
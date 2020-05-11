
import time
from collections import defaultdict
import logging
from tqdm import tqdm

# from torchvision.datasets.folder import default_loader
from data.transforms.image import cv2_loader, BGR2Lab
import faiss
import numpy as np
from scipy.spatial.distance import cdist
from numpy import linalg as LA
import os

# __all__ = ['Kmeans']

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    Dist, Ind = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    # if verbose:
        # print('k-means loss evolution: {0}'.format(losses))

    centers = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, -1)
    kmeans_trans_weight = -2 * centers
    kmeans_trans_bias = (centers**2).sum(axis=1)  # (K,)

    return [int(n[0]) for n in Ind], [float(n[0]) for n in Dist], losses[-1], clus, \
        [kmeans_trans_weight, kmeans_trans_bias], centers


class Kmeans:
    def __init__(self, k):
        self.k = k

        self.kmeans_trans = None
        self.centroids = None

    def train(self, data, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()
        # cluster the data
        I, D, loss, clus, self.kmeans_trans, centers = run_kmeans(data.astype(np.float32), self.k, verbose)
        self.centroids = centers
        self.images_lists = defaultdict(list)
        self.dist_lists = defaultdict(list)
        for i, (cls, dist) in enumerate(zip(I, D)):
            self.images_lists[cls].append(i)
            self.dist_lists[cls].append(dist)
        for k, v in self.images_lists.items():
            idx = np.argsort(self.dist_lists[k])
            self.images_lists[k] = [v[i] for i in idx]

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))
        return loss

def samplefeats(image_path, size, N=10000, P=10):
    selected_paths = np.random.choice(image_path, N//P, replace=False)
    out = []
    print('sampling imgs for clustering LAB')
    for i in tqdm(selected_paths):
        img = np.asarray(BGR2Lab(cv2_loader(i, size)))
        assert img.shape[-1] == 3
        h, w = img.shape[0], img.shape[1]
        idx = np.random.choice(h * w, P)
        for j in idx:
            out.append(img[j // w, j % w])
    out = np.vstack(out)
    assert out.shape[0] == N
    return out

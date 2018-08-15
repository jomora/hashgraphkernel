# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre

from auxiliarymethods import auxiliary_methods as aux
from setuptools.dist import Feature


def hash_graph_kernel(graph_db, base_kernel, kernel_parameters, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()

    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes])
    offset = 0


    # Get attributes from all graph instances
    graph_indices = []
    for g in graph_db:
        for i, v in enumerate(g.vertices()):
            colors_0[i + offset] = g.vp.na[v]

        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    # Normalize attributes: center to the mean and component wise scale to unit variance
    if scale_attributes:
        colors_0 = pre.scale(colors_0, axis=0)

    feature_vectors,gram_matrix = compute_feature_vectors(iterations,colors_0,dim_attributes,lsh_bin_width,sigma,base_kernel,kernel_parameters,use_gram_matrices,graph_db)

    if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()

    if normalize_gram_matrix:
        gram_matrix = aux.normalize_gram_matrix(gram_matrix)
    return gram_matrix,feature_vectors

def compute_feature_vectors(iterations,colors_0,dim_attributes,lsh_bin_width,sigma,base_kernel,kernel_parameters,use_gram_matrices,graph_db):
    n = len(graph_db)
    gram_matrix = sparse.lil_matrix((n,n),dtype=np.float64).tocsr() #np.zeros([n, n])
    for it in xrange(0, iterations):
        colors_hashed = aux.locally_sensitive_hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma)
        print(colors_hashed, colors_0, dim_attributes,lsh_bin_width,sigma)
        tmp = base_kernel(graph_db, colors_hashed, *kernel_parameters)
        if it == 0 and not use_gram_matrices:
            feature_vectors = tmp
        else:
            if use_gram_matrices:
                feature_vectors = tmp
                feature_vectors = feature_vectors.tocsr()
                feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
                gram_matrix += feature_vectors.dot(feature_vectors.T)
            else:
                feature_vectors = sparse.hstack((feature_vectors, tmp))
    return feature_vectors.tocsr(),gram_matrix

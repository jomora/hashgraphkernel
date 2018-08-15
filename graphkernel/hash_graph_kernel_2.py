# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre

from auxiliarymethods import auxiliary_methods as aux
from setuptools.dist import Feature
from auxiliarymethods.logging import format_time, time_it
import time

def hash_graph_kernel(graph_db, base_kernel, kernel_parameters, hashing, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    print ("# Starting hgk sequential at " + format_time(time.time()))

    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    n = len(graph_db)

    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes])
    offset = 0

    gram_matrix = sparse.lil_matrix((n,n),dtype=np.float64).tocsr() #np.zeros([n, n])

    # Get attributes from all graph instances
    graph_indices = []
    for g in graph_db:
        for i, v in enumerate(g.vertices()):
            colors_0[i + offset] = g.vp.na[v]

        offset += g.num_vertices()
        graph_indices.append((offset, offset + g.num_vertices() - 1))

    # Normalize attributes: center to the mean and component wise scale to unit variance
    if scale_attributes:
        colors_0 = pre.scale(colors_0, axis=0)

    if not use_gram_matrices:
        print ("# Starting loop at " + format_time(time.time()))
        for it in xrange(0, iterations):
            colors_hashed = hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma)

            tmp = base_kernel(graph_db, colors_hashed, *kernel_parameters)
            if it == 0:
                feature_vectors = tmp
            else:
                feature_vectors = sparse.hstack((feature_vectors, tmp))

        feature_vectors = feature_vectors.tocsr()
        print ("# Ending loop at " + format_time(time.time()))

        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()
    else: # if use_gram_matrices
        print ("# Starting loop at " + format_time(time.time()))
        for it in xrange(0, iterations):
            colors_hashed = hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma)
            tmp = base_kernel(graph_db, colors_hashed, *kernel_parameters)
            feature_vectors = tmp
            feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
            gram_matrix += feature_vectors.dot(feature_vectors.T)

        print ("# Ending loop at " + format_time(time.time()))

    print ("# Start normalizing gram at " + format_time(time.time()))
    if normalize_gram_matrix:
        gram_matrix = aux.normalize_gram_matrix(gram_matrix)
    print ("# End normalizing gram at " + format_time(time.time()))

    print ("# Ending hgk sequential at " + format_time(time.time()))
    return gram_matrix,feature_vectors

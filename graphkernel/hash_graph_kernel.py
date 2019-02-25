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
import scipy
import datetime
DEBUG = False

def logNow(): return "[" + datetime.datetime.now().replace(microsecond=0).isoformat() + "]"


def hash_graph_kernel(LOG, graph_db, base_kernel, kernel_parameters, hashing, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    start = time.time()
    LOG.info(logNow() + " [HGK] \033[1;32m# Starting hgk sequential at " + format_time(start) + "\033[0;37m")

    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    n = len(graph_db)

    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes],dtype=np.float64)
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

    loop_start = time.time()
    LOG.debug(logNow() + " [HGK] # Using gram matrix " + format_time(loop_start))
    LOG.debug(logNow() + " [HGK] # Starting loop at " + format_time(loop_start))
    for it in range(0, iterations):
        colors_hashed = hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma)

        tmp = base_kernel(graph_db, colors_hashed, *kernel_parameters)
        if it == 0 and not use_gram_matrices:
            feature_vectors = tmp
        else:
            if use_gram_matrices:
                feature_vectors = tmp
                # feature_vectors = feature_vectors.tocsr()
                feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
                gram_matrix += feature_vectors.dot(feature_vectors.T)
            else:
                feature_vectors = sparse.hstack((feature_vectors, tmp))

    if scipy.sparse.issparse(feature_vectors):
        feature_vectors = feature_vectors.tocsr()
    loop_end = time.time()
    LOG.debug(logNow() + " [HGK] # Ending loop at " + format_time(loop_end))
    LOG.info(logNow() + " [HGK] \033[1;32m# Duration of loop in [s]: " + str(loop_end - loop_start) + "\033[0;37m")
    if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()

    LOG.debug(logNow() + " [HGK] # Start normalizing gram at " + format_time(time.time()))
    if normalize_gram_matrix:
        gram_matrix = aux.normalize_gram_matrix(gram_matrix)
        LOG.debug(logNow() + " [HGK] # End normalizing gram at " + format_time(time.time()))

    end = time.time()
    LOG.debug(logNow() + " [HGK] # Ending hgk sequential at " + format_time(end))
    LOG.info(logNow() + " [HGK] \033[1;32m# Duration of hgk sequential in [s]: " + str(end - start) + "\033[0;37m")
    return gram_matrix,feature_vectors

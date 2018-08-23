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
from multiprocessing import Pool
import os
from scipy import sparse as spa

DEBUG = False


def hash_graph_kernel(LOG, graph_db, base_kernel, kernel_parameters, hashing, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    start = time.time()
    LOG.info("\033[1;32m# Starting hgk parallel at " + format_time(start) + "\033[0;37m")

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
    pool = Pool()
    LOG.info("\033[1;32m# Using " + str(pool._processes) + " processes\033[0;37m")

    if not use_gram_matrices:
        loop_start = time.time()

        LOG.debug("# Not using GRAM at " + format_time(loop_start))
        LOG.debug("# Starting loop at " + format_time(loop_start))
        TASKS = [
            (base_kernel,
            iterations,
            graph_db, hashing(
                colors_0,
                dim_attributes,
                lsh_bin_width,
                sigma=sigma),
                kernel_parameters)
                for i in range(0,iterations)]

        results = pool.map_async(compute_feature_vectors_parallel, TASKS,chunksize=1)

        pool.close()
        pool.join()

            # if it == 0:
            #     feature_vectors = tmp
            # else:
            #     feature_vectors = sparse.hstack((feature_vectors, tmp))
        results = np.asarray([r for r in results.get()])
        feature_vectors = spa.hstack(results)

        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        loop_end = time.time()
        LOG.debug("# Ending loop at " + format_time(loop_end))
        LOG.info ("\033[1;32m# Duration of loop in [s]: " + str(loop_end - loop_start) + "\033[0;37m")
        #gram_matrix = gram_matrix.toarray()
    else: # if use_gram_matrices
        loop_start = time.time()

        LOG.debug ("# Using gram matrix " + format_time(loop_start))
        LOG.debug("# Starting loop at " + format_time(loop_start))

        TASKS = [
            (base_kernel,
            iterations,
            graph_db, hashing(
                colors_0,
                dim_attributes,
                lsh_bin_width,
                sigma=sigma),
                kernel_parameters)
        for i in range(0,iterations)]

        results = pool.map_async(compute_gram_parallel, TASKS,chunksize=1)

        pool.close()
        pool.join()
        for i,res in enumerate(results.get()):
            LOG.debug("Getting result " + str(i))
            gram_matrix += res
        # for it in xrange(0, iterations):
        #     colors_hashed = hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma)
        #     gram_matrix += compute_gram_parallel((base_kernel, iterations, graph_db, colors_hashed, kernel_parameters))
        feature_vectors = []
        loop_end = time.time()
        LOG.debug("# Ending loop at " + format_time(loop_end))
        LOG.info("\033[1;32m# Duration of loop in [s]: " + str(loop_end - loop_start) + "\033[0;37m")


    LOG.debug("# Start normalizing gram at " + format_time(time.time()))
    if normalize_gram_matrix:
        gram_matrix = aux.normalize_gram_matrix(gram_matrix)
        LOG.debug("# End normalizing gram at " + format_time(time.time()))

    end = time.time()
    LOG.debug("# Ending hgk parallel at " + format_time(end))
    LOG.info("\033[1;32m# Duration of hgk parallel in [s]: " + str(end - start) + "\033[0;37m")

    return gram_matrix,feature_vectors

def compute_gram_parallel(args):
    # LOG.debug( "# Starting process ",os.getpid()," at ",format_time(time.time()))
    base_kernel, iterations, graph_db, colors_hashed, kernel_parameters = args
    feature_vectors = base_kernel(graph_db, colors_hashed, *kernel_parameters)
    feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
    return feature_vectors.dot(feature_vectors.T)

def compute_feature_vectors_parallel(args):
    # LOG.debug("# Starting process ",os.getpid()," at ",format_time(time.time()))
    base_kernel, iterations, graph_db, colors_hashed, kernel_parameters = args
    return spa.bsr_matrix(base_kernel(graph_db, colors_hashed, *kernel_parameters)).tocsr()

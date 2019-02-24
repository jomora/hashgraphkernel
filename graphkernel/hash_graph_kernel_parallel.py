# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
import scipy
from sklearn import preprocessing as pre
from auxiliarymethods.logging import format_time, time_it
import time
import os
import datetime
from auxiliarymethods import auxiliary_methods as aux
from setuptools.dist import Feature
import multiprocessing
from multiprocessing import Process,Queue,Pool
import sys

def logNow(): return "[" + datetime.datetime.now().replace(microsecond=0).isoformat() + "]"
    
def run_base_kernel_parallel(tup):
    print(logNow() + " [HGK] [WL-PAR] # Starting process " + str(os.getpid()) + " at " + format_time(time.time()))
    base_kernel, args = tup
    graph_db, hashed_attributes, kwargs = args
    tmp = base_kernel(graph_db, hashed_attributes, *kwargs)
    print(logNow() + " [HGK] [WL-PAR] # Returning process " + str(os.getpid()) + " at " + format_time(time.time()))

    return tmp

def hash_graph_kernel_parallel(graph_db, base_kernel, kernel_parameters, hashing, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    print(logNow() + " [HGK] [WL-PAR] # Starting hgk parallel at " + format_time(time.time()))

    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    n = len(graph_db)

    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes],dtype=np.float32)
    offset = 0

    gram_matrix = sparse.lil_matrix((n,n),dtype=np.float32).tocsr()

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

    # Create hashed colors in advance for every iteration
    # colors_hashed_list = [hashing(colors_0,
    #                             dim_attributes,
    #                             lsh_bin_width,
    #                             sigma=sigma)
    #             for it in xrange(0, iterations)]


    queues_and_processes = []
    cores = multiprocessing.cpu_count()
    numProcesses = min(cores,10)
    print(logNow() + " [HGK] [WL-PAR] # Running parallel with %d cores" % numProcesses)
    pool = Pool(processes=numProcesses)


    TASKS = [(base_kernel,(graph_db, hashing(colors_0,
                                dim_attributes,
                                lsh_bin_width,
                                sigma=sigma), kernel_parameters)) for i in xrange(0,iterations)]

    print(logNow() + " [HGK] [WL-PAR] # Starting pool at " + format_time(time.time()))
    results = pool.map_async(run_base_kernel_parallel, TASKS,chunksize=1)

    pool.close()
    pool.join()

    print(logNow() + " [HGK] [WL-PAR] # Joining pool at " + format_time(time.time()))
    # results.sort(key=lambda tup: tup[0])
    # results = [tup[1] for tup in results]
    results = results.get()
    print(logNow() + " [HGK] [WL-PAR] # Starting loop at " + format_time(time.time()))
    for it in xrange(0, iterations):
    #for tmp in results:
        tmp = results[i]
        if it == 0 and not use_gram_matrices:
            feature_vectors = tmp
        else:
            if use_gram_matrices:
                feature_vectors = tmp
                feature_vectors = feature_vectors.tocsr() if scipy.sparse.issparse(feature_vectors) else feature_vectors
                feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
                gram_matrix += feature_vectors.dot(feature_vectors.T) #.toarray()
            else:
                feature_vectors = sparse.hstack((feature_vectors, tmp))

    feature_vectors = feature_vectors.tocsr() if scipy.sparse.issparse(feature_vectors) else feature_vectors
    print(logNow() + " [HGK] [WL-PAR] # Finishing loop at " + format_time(time.time()))

    if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()
    print (logNow() + " [HGK] [WL-PAR] # Start normalizing gram at " + format_time(time.time()))
    if normalize_gram_matrix:
        gram_matrix = time_it(aux.normalize_gram_matrix,gram_matrix)
    print (logNow() + " [HGK] [WL-PAR] # End normalizing gram at " + format_time(time.time()))
    print (logNow() + " [HGK] [WL-PAR] # Ending hgk parallel at " + format_time(time.time()))

    return gram_matrix,feature_vectors

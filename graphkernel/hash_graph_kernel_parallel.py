# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre
from auxiliarymethods.logging import format_time, time_it
import time
import os
from auxiliarymethods import auxiliary_methods as aux
from setuptools.dist import Feature

def run_base_kernel_parallel(tup):
    print "# Starting process ",os.getpid()," at ",format_time(time.time())
    base_kernel, args = tup
    graph_db, hashed_attributes, kwargs = args
    tmp = base_kernel(graph_db, hashed_attributes, *kwargs)
    print "# Returning process ",os.getpid()," at ",format_time(time.time())

    return tmp

def hash_graph_kernel_parallel(graph_db, base_kernel, kernel_parameters, hashing, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
    print ("# Starting hgk parallel at " + format_time(time.time()))

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



    from multiprocessing import Process,Queue,Pool
    queues_and_processes = []
    pool = Pool(processes=10)


    TASKS = [(base_kernel,(graph_db, hashing(colors_0,
                                dim_attributes,
                                lsh_bin_width,
                                sigma=sigma), kernel_parameters)) for i in xrange(0,iterations)]

    print ("# Starting pool at " + format_time(time.time()))
    results = pool.map_async(run_base_kernel_parallel, TASKS,chunksize=1)

    pool.close()
    pool.join()
    print ("# Joining pool at " + format_time(time.time()))
    # results.sort(key=lambda tup: tup[0])
    # results = [tup[1] for tup in results]
    results = results.get()
    print ("# Starting loop at " + format_time(time.time()))
    for it in xrange(0, iterations):
    #for tmp in results:
        tmp = results[i]
        if it == 0 and not use_gram_matrices:
            feature_vectors = tmp
        else:
            if use_gram_matrices:
                print("ping")
                feature_vectors = tmp
                feature_vectors = feature_vectors.tocsr()
                feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
                print(type(gram_matrix))
                gram_matrix += feature_vectors.dot(feature_vectors.T) #.toarray()
                print(type(gram_matrix))
                print()
            else:
                print("pong")
                feature_vectors = sparse.hstack((feature_vectors, tmp))

    feature_vectors = feature_vectors.tocsr()
    print ("# Finishing loop at " + format_time(time.time()))

    if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()
    from auxiliarymethods.logging import time_it
    print ("# Start normalizing gram at " + format_time(time.time()))
    if normalize_gram_matrix:
        gram_matrix = time_it(aux.normalize_gram_matrix,gram_matrix)
    print ("# End normalizing gram at " + format_time(time.time()))
    print ("# Ending hgk parallel at " + format_time(time.time()))

    return gram_matrix,feature_vectors

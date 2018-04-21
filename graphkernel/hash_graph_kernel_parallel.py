# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import math as m
import numpy as np
import scipy.sparse as sparse
from sklearn import preprocessing as pre

from auxiliarymethods import auxiliary_methods as aux
from setuptools.dist import Feature


def hash_graph_kernel_parallel(graph_db, base_kernel, kernel_parameters, iterations=20, lsh_bin_width=1.0, sigma=1.0,
                      normalize_gram_matrix=True, use_gram_matrices=False, scale_attributes=True):
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
    colors_hashed_list = [aux.locally_sensitive_hashing(colors_0, dim_attributes, lsh_bin_width, sigma=sigma) for it in xrange(0, iterations)]

    def run_base_kernel_parallel(queue,graph_db, hashed_attributes, kwargs):
        tmp = base_kernel(graph_db, hashed_attributes, *kwargs)
        queue.put(tmp)

    from multiprocessing import Process,Queue,Pool
    queues_and_processes = []
    pool = Pool(processes=2) 
       
    for i in xrange(0,iterations):
        q = Queue()
        p = Process(target=run_base_kernel_parallel, args=(q,graph_db,colors_hashed_list[i],kernel_parameters))
        p.start()
        queues_and_processes.append((q,p))
        #tmp = base_kernel(graph_db, colors_hashed, *kernel_parameters)

    # Gather data and terminate processes
    results = []
    for q,p in queues_and_processes:
        results.append(q.get())
        p.join()
        

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

    if not use_gram_matrices:
        # Normalize feature vectors
        feature_vectors = m.sqrt(1.0 / iterations) * (feature_vectors)
        # Compute Gram matrix
        gram_matrix = feature_vectors.dot(feature_vectors.T)
        #gram_matrix = gram_matrix.toarray()
    from auxiliarymethods.logging import time_it
    if normalize_gram_matrix:
        gram_matrix = time_it(aux.normalize_gram_matrix,gram_matrix)

    return gram_matrix,feature_vectors

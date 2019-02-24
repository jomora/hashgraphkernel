
# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods.dataset_parsers import DatasetParser
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from graphkernel import wl_kernel_sparse as wl_sparse
import numpy as np
import time
import datetime
from auxiliarymethods.logging import format_time, time_it
from graphkernel import hash_graph_kernel_parallel as rbk_parallel
import logging

import argparse
import scipy.sparse as sps

import multiprocessing
from multiprocessing import Process,Queue,Pool

import sys
import os

def logNow(): return "[" + datetime.datetime.now().replace(microsecond=0).isoformat() + "]"
    
def main(dataset,basepath,parallel,compute_feature_vectors):


    print(logNow() + " [HGK] # Dataset: " + dataset)
    print(logNow() + " [HGK] # Base directory: " + basepath)
    print(logNow() + " [HGK] # Running in parallel mode: " + str(parallel))
    start = time.time()
    print(logNow() + " [HGK] # Program started at " + format_time(start))
    #dataset = "all"
    #dataset = "ENZYMES"
    # PATH = os.environ['SEML_DATA'] + '/output/'
    # PATH = "datasets/"

    dp = DatasetParser(basepath)

    print(logNow() + " [HGK] # Processing dataset: " + dataset)
    # Load ENZYMES data set
    # graph_db, classes = time_it(dp.read_txt,(dataset))
    graph_db = time_it(dp.read_graph_db,dataset)

    # Parameters used:
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_sp = [False, True, False]

    # Parameters used:
    # Number of iterations for WL: 3
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_wl = [3, False, True, False]

    # Use discrete labels, too
    # kernel_parameters_sp = [False, False, 1]
    # kernel_parameters_wl = [3, False, False, 1]


    # Compute gram matrix for HGK-WL
    # 20 is the number of iterations
#     gram_matrix, feature_vectors = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
#                                         scale_attributes=True, lsh_bin_width=1.0, sigma=1.0, use_gram_matrices=True)
#     # Normalize gram matrix
#     gram_matrix = aux.normalize_gram_matrix(gram_matrix)
#    gram_matrix = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
 #                                       scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
    # Normalize gram matrix
  #  gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Compute gram matrix for HGK-SP
    # 20 is the number of iterations
    LOG = logging
    if parallel:
        gram_matrix, feature_vectors = time_it(rbk_parallel.hash_graph_kernel_parallel,graph_db, wl_sparse.weisfeiler_lehman_subtree_kernel,
            kernel_parameters_wl, aux.locally_sensitive_hashing, iterations=10, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0, use_gram_matrices=True,normalize_gram_matrix=False)
    else:
        gram_matrix, feature_vectors = time_it(rbk.hash_graph_kernel,LOG, graph_db, wl_sparse.weisfeiler_lehman_subtree_kernel,
            kernel_parameters_wl, aux.locally_sensitive_hashing, iterations=10, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0, use_gram_matrices=True,normalize_gram_matrix=False)
    
    print (logNow() + " [HGK] Shape of feature vectors: " + str(feature_vectors.shape))

    # Write out LIBSVM matrix
    #dp.write_lib_svm(gram_matrix, classes, "gram_matrix")

    # Write out simple Gram matrix used for clustering
    # time_it(dp.write_gram_matrix,gram_matrix, dataset)
    # Write out simple Gram matrix in sparse format

    print(logNow() + " [HGK] Gram matrix in NPZ format")
    time_it(dp.write_sparse_gram_matrix,gram_matrix.tocoo(),dataset)
    print(logNow() + " [HGK] Shape of Gram Matrix: " + str(np.shape(gram_matrix)))

    print(logNow() + " [HGK] Feature vectors in NPZ format")
    time_it(dp.write_sparse_feature_vectors,sps.csr_matrix(feature_vectors),dataset)
    print(logNow() + " [HGK] Shape of Gram Matrix: " + str(np.shape(gram_matrix)))

    print(logNow() + " [HGK] Feature vectors in NPZ format")
    time_it(dp.write_sparse_feature_vectors,sps.csr_matrix(feature_vectors),dataset)
    print(logNow() + " [HGK] Shape of feature vectors: " + str(np.shape(feature_vectors)))

    time_it(dp.write_gram_matrix,gram_matrix.todense(),dataset)
    time_it(dp.write_feature_vectors,feature_vectors,dataset)

    #dp.write_feature_vectors(feature_vectors, dataset, [])
    end = time.time()
    print (logNow() + " [HGK] # Program ended at ") + format_time(start)
    print (logNow() + " [HGK] # Duration in [s]: ") + str(end - start)


def read_args():
    parser = argparse.ArgumentParser(description='Run the  Hash Graph Kernel')
    parser.add_argument('-ds','--dataset', action="store", dest="dataset",
        help="The name of the dataset (must match the directory" \
            " name in which the datset is stored, and the prefix" \
            " of all files in the dataset directors.")
    parser.add_argument('-b', '--base',action="store", dest="base",
        help="The base directory in which the datset directory is located")
    parser.add_argument('-p','--parallel', action="store_true",dest='parallel', default=False)
    parser.add_argument('-V', '--feature-vectors',action="store_true", dest="feature_vectors",
        help="If present then feature vectors will be stored.")
    args = parser.parse_args()
    problem = False
    if args.dataset == None:
        print(logNow() + " [HGK] # No dataset given")
        problem = True
    if args.base == None:
        print(logNow() + " [HGK] # No base directory given")
        problem = True
    if problem:
        print(logNow() + " [HGK] --> Argument error: exiting")
        exit(0)
    dataset = args.dataset
    basepath = args.base
    parallel = args.parallel
    compute_feature_vectors = args.feature_vectors
    return dataset,basepath,parallel,compute_feature_vectors

if __name__ == "__main__":
    dataset,basepath,parallel,compute_feature_vectors = read_args()
    main(dataset,basepath,parallel,compute_feature_vectors)

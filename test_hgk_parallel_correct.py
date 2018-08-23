from graphkernel import hash_graph_kernel as rbk
from graphkernel import hash_graph_kernel_2 as rbk_2
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods import auxiliary_methods as aux

import graph_tool as gt
import numpy as np
import itertools
import pprint
import logging


def test_parallel_vs_sequential():

    # dataset = "qualitas-corpus"
    # dir = "/home/jomora/seml/data/parallel-test/"
    dataset = "ENZYMES"
    dir = "datasets/"

    LOG = logging.Logger(name=__file__,level=logging.INFO)
    fh = logging.FileHandler(dir + dataset + "/output.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    LOG.addHandler(fh)
    LOG.info("\033[1;32m############### STARTING TEST ############### \033[0;37m")

    dp = DatasetParser(dir)
    graph_db = dp.read_graph_db(dataset)
    # colors_0 = np.zeros((np.sum([len(list(g.vertices())) for g in graph_db]),),dtype=np.int64)
    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    g = graph_db[0]
    v = list(graph_db[0].vertices())[0]
    dim_attributes = len(g.vp.na[v])
    colors_0 = np.zeros([num_vertices, dim_attributes])
    offset = 0
    graph_indices = []
    for g in graph_db:
        for i, v in enumerate(g.vertices()):
            colors_0[i + offset] = g.vp.na[v]

        graph_indices.append((offset, offset + g.num_vertices() - 1))
        offset += g.num_vertices()

    kernel_parameters_wl = [1, True, True, 0]

    kernel_iterations = 10
    hashes = [aux.locally_sensitive_hashing(colors_0, dim_attributes, 1.0, sigma=1.0) for i in range(kernel_iterations)]

    def create_hashes():
        hashes = [aux.locally_sensitive_hashing(colors_0, dim_attributes, 1.0, sigma=1.0) for i in range(kernel_iterations)]
        return iter(hashes),iter(hashes)

    def create_hash_function(hashes):
        def hashing(m, d, w, sigma=1.0):
            return next(hashes)
        return hashing


    LOG.info("\033[1;32m# use_gram_matrices=True, normalize_gram_matrix=True: \033[0;37m")
    hashes_1, hashes_2 = create_hashes()
    gram_matrix_1, _ = rbk_2.hash_graph_kernel(LOG,graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, create_hash_function(hashes_2 ), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)
    gram_matrix_2, _ = rbk.hash_graph_kernel(LOG, graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,create_hash_function(hashes_1), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0
    del gram_matrix_1
    del gram_matrix_2

    LOG.info("\033[1;32m# use_gram_matrices=True, normalize_gram_matrix=True: \033[0;37m")
    hashes_1, hashes_2 = create_hashes()
    gram_matrix_1, _ = rbk_2.hash_graph_kernel(LOG,graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, create_hash_function(hashes_2 ), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=False)
    gram_matrix_2, _ = rbk.hash_graph_kernel(LOG, graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,create_hash_function(hashes_1), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=False)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0
    del gram_matrix_1
    del gram_matrix_2

    LOG.info("\033[1;32m# use_gram_matrices=False, normalize_gram_matrix=True: \033[0;37m")
    hashes_1, hashes_2 = create_hashes()
    gram_matrix_1, _ = rbk_2.hash_graph_kernel(LOG,graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, create_hash_function(hashes_2 ), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=False,normalize_gram_matrix=True)
    gram_matrix_2, _ = rbk.hash_graph_kernel(LOG, graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,create_hash_function(hashes_1), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=False,normalize_gram_matrix=True)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0
    del gram_matrix_1
    del gram_matrix_2

    LOG.info("\033[1;32m# use_gram_matrices=False, normalize_gram_matrix=False: \033[0;37m")
    hashes_1, hashes_2 = create_hashes()
    gram_matrix_1, _ = rbk_2.hash_graph_kernel(LOG,graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, create_hash_function(hashes_2 ), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=False,normalize_gram_matrix=False)
    gram_matrix_2, _ = rbk.hash_graph_kernel(LOG, graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,create_hash_function(hashes_1), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=False,normalize_gram_matrix=False)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0
    del gram_matrix_1
    del gram_matrix_2

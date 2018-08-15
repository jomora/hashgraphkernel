from graphkernel import hash_graph_kernel as rbk
from graphkernel import hash_graph_kernel_parallel as rbk_parallel
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods import auxiliary_methods as aux

import graph_tool as gt
import numpy as np
import itertools
import pprint

def test_rbk_enzymes():
    dataset = "ENYZMES"
    dp = DatasetParser("./datasets/")
    graph_db = dp.read_graph_db("ENZYMES")
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

    kernel_iterations = 2
    hashes = [aux.locally_sensitive_hashing(colors_0, dim_attributes, 1.0, sigma=1.0) for i in range(kernel_iterations)]


    hashes_1 = iter(hashes)
    hashes_2 = iter(hashes)
    def create_hash_function(hashes):
        def hashing(m, d, w, sigma=1.0):
            return hashes.next()
        return hashing


    gram_matrix_1, feature_vectors_1 = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, create_hash_function(hashes_2 ), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)
    gram_matrix_2, feature_vectors_2 = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,create_hash_function(hashes_1), kernel_iterations, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0

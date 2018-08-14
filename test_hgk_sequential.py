from graphkernel import hash_graph_kernel as rbk
from graphkernel import hash_graph_kernel_parallel as rbk_parallel
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods import auxiliary_methods as aux

import graph_tool as gt
import numpy as np


def test_rbk_enzymes():
    dataset = "ENYZMES"
    dp = DatasetParser("./datasets/")
    graph_db = dp.read_graph_db("ENZYMES")
    colors_hashed = np.zeros((np.sum([len(list(g.vertices())) for g in graph_db]),),dtype=np.int64)
    kernel_parameters_wl = [1, True, True, 0]

    def hashing(m, d, w, sigma=1.0):
        return colors_hashed

    hash_function = hashing #aux.locally_sensitive_hashing
    gram_matrix_1, feature_vectors_1 = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl, hash_function, 1, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)
    gram_matrix_2, feature_vectors_2 = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel,
        kernel_parameters_wl,hash_function, 1, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0,
        use_gram_matrices=True,normalize_gram_matrix=True)

    assert np.sum(gram_matrix_1 - gram_matrix_2) == 0.0

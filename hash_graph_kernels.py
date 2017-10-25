# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl

ALGORITHMS = "ALGORITHMS"
ENZYMES = "ENZYMES"

def main():
    # Load ENZYMES data set
    # graph_db, classes = dp.read_txt("ENZYMES")
    graph_db = dp.read_graph_db(ENZYMES)
    # Parameters used: 
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_sp = [False, False, 0]

    # Parameters used: 
    # Number of iterations for WL: 3
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_wl = [3, False, False, 0]

    # Use discrete labels, too
    # kernel_parameters_sp = [False, False, 1]
    # kernel_parameters_wl = [3, False, False, 1]


    # Compute gram matrix for HGK-WL
    # 20 is the number of iterations
    gram_matrix = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
                                        scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
    # Normalize gram matrix
    gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Compute gram matrix for HGK-SP
    # 20 is the number of iterations
    gram_matrix = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel, kernel_parameters_wl, 20,
                                        scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

    # Normalize gram matrix
    gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Write out LIBSVM matrix
    # dp.write_lib_svm(gram_matrix, classes, "gram_matrix")
    dp.write_gram_matrix(gram_matrix,"gram_matrix_simple")


if __name__ == "__main__":
    main()

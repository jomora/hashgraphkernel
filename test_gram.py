
from graphkernel import wl_kernel as wl
import numpy as np
from auxiliarymethods import dataset_parsers as dp
import scipy.sparse as sps

def test():
    kernel_parameters_wl = [3, False, False, 0]
    graph_db, classes = dp.read_txt("ENZYMES")
    num_vertices = 0
    for g in graph_db:
        num_vertices += g.num_vertices()
    colors = np.ones((num_vertices,),dtype=np.int64)
    gram_1 =  wl.weisfeiler_lehman_subtree_kernel(graph_db,colors,*kernel_parameters_wl)
    gram_2 =  wl.weisfeiler_lehman_subtree_kernel(graph_db,colors,*kernel_parameters_wl)
    print(np.sum(gram_1 - gram_2))


def read_gram(file_name):
    with open(file_name,'r') as f:
        lines = list(f)
        length = len(lines)
        gram = np.zeros((length,length)) 
           
        for i,row in enumerate(lines):
            for j,col in enumerate(row.strip().split(' ')):
                gram[i][j] = float(col)
    return gram


def main():
    file_a = "datasets/ENZYMES/ENZYMES_sparse_gram_sequential.npz"
    file_b = "datasets/ENZYMES/ENZYMES_sparse_gram_parallel.npz"
    seq = sps.load_npz(file_a) #read_gram("datasets/ENZYMES/ENZYMES_gram_matrix_simple_sequential")
    par = sps.load_npz(file_b) #read_gram("datasets/ENZYMES/ENZYMES_gram_matrix_simple_parallel")
    print(np.sum(seq-par))
    test()
    
if __name__ == "__main__":
    main()
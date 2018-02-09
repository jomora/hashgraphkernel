
from graphkernel import wl_kernel as wl
import numpy as np
from auxiliarymethods import dataset_parsers as dp

kernel_parameters_wl = [3, False, False, 0]
graph_db, classes = dp.read_txt("ENZYMES")
num_vertices = 0
for g in graph_db:
    num_vertices += g.num_vertices()
colors = np.ones((num_vertices,),dtype=np.int64)
gram_1 =  wl.weisfeiler_lehman_subtree_kernel(graph_db,colors,*kernel_parameters_wl)
gram_2 =  wl.weisfeiler_lehman_subtree_kernel(graph_db,colors,*kernel_parameters_wl)
print(np.sum(gram_1 - gram_2))

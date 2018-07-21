from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
import graph_tool as gt
import numpy as np
from pprint import pprint

def test_wl_kernel():
    g = create_graph()
    graph_db = [g,g]
    colors_hashed = np.zeros((np.sum([len(list(g.vertices())) for g in graph_db]),),dtype=np.int64)
    # Parameters used:
    # Number of iterations for WL: 1
    # Compute gram matrix: False,
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_wl = [1, True, True, 0]

    gram_1 = wl.weisfeiler_lehman_subtree_kernel(graph_db,colors_hashed,*kernel_parameters_wl)
    gram_2 = wl.weisfeiler_lehman_subtree_kernel(graph_db,colors_hashed,*kernel_parameters_wl)

    pprint(np.sum(gram_1 -gram_2))
    assert np.sum(gram_1 -gram_2) == 0.0

def create_graph():
    g = gt.Graph(directed=False)

    v_neighbors = g.new_vertex_property("int")

    v1 = g.add_vertex()
    v2 = g.add_vertex()
    v3 = g.add_vertex()
    v4 = g.add_vertex()
    v5 = g.add_vertex()
    v6 = g.add_vertex()

    edges = [(v1,v2),
            (v2,v3),
            (v2,v4),
            (v3,v5),
            (v4,v5),
            (v5,v6)]
    for a,b in edges:
        g.add_edge(a,b)
        g.add_edge(b,a)

    for v in g.vertices():
        v_neighbors[v] = len(list(v.out_neighbours()))


    assert len(list(v1.out_neighbours())) == 2
    assert len(list(v2.out_neighbours())) == 6
    assert len(list(v3.out_neighbours())) == 4
    assert len(list(v4.out_neighbours())) == 4
    assert len(list(v5.out_neighbours())) == 6
    assert len(list(v6.out_neighbours())) == 2

    return g

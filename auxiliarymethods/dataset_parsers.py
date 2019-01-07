# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

import graph_tool as gt
import numpy as np
import os.path as path
import os 
import scipy.sparse as sps
import csv

import itertools

class DatasetParser:
    def __init__(self, path):
        self.PATH = path


    def read_txt(self,ds_name):
        return self.read_graph_db(ds_name), self.read_classes(ds_name)

    def read_graph_db(self,ds_name):
        pre = ""
        print("# Creating...")
        print("# \t- graph_indicator")
        with open(self.PATH + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r") as f:
            graph_indicator = [int(i) - 1 for i in list(f)]
        f.closed
        xrange = lambda stop: iter(itertools.count().next, stop)
        # Nodes
        num_graphs = max(graph_indicator)
        node_indices = []
        offset = []
        c = 0
        print("# \t- node_indices")
        for i in xrange(num_graphs + 1):
            if i % 1000 == 0:
                print("#\t\t- iteration %d of %d" % (i,num_graphs + 1))
            offset.append(c)
            c_i = graph_indicator.count(i)
            node_indices.append((c, c + c_i - 1))
            c += c_i

        print("# \t- graph_db")
        graph_db = []
        vertex_list = []
        for i in node_indices:
            g = gt.Graph(directed=False)
            vertex_list_g = []
            for _ in xrange(i[1] - i[0] + 1):
                vertex_list_g.append(g.add_vertex())

            graph_db.append(g)
            vertex_list.append(vertex_list_g)

        # Edges
        print("# \t- edges")
        with open(self.PATH + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
            edges = [i.split(',') for i in list(f)]
        f.closed

        edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

        edge_indicator = []
        edge_list = []
        for e in edges:
            g_id = graph_indicator[e[0]]
            edge_indicator.append(g_id)
            g = graph_db[g_id]
            off = offset[g_id]

            # Avoid multigraph
            if not g.edge(e[0] - off, e[1] - off):
                edge_list.append(g.add_edge(e[0] - off, e[1] - off))
        # Node labels
        if path.exists(self.PATH + pre + ds_name + "/" + ds_name + "_node_labels.txt"):
            print("# \t- node_labels")
            with open(self.PATH + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r") as f:
                node_labels = [int(i) for i in list(f)]
            f.closed

            i = 0
            for g in graph_db:
                g.vp.nl = g.new_vertex_property("int")
                for v in g.vertices():
                    g.vp.nl[v] = node_labels[i]
                    i += 1


        # Node Attributes
        if path.exists(self.PATH + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
            print("# \t- node_attributes")
            with open(self.PATH + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
                node_attributes = [map(float, i.split(',')) for i in list(f)]
            f.closed

            i = 0
            for g in graph_db:
                g.vp.na = g.new_vertex_property("vector<float>")
                for v in g.vertices():
                    g.vp.na[v] = node_attributes[i]
                    i += 1


        # Edge Labels
        if path.exists(self.PATH + ds_name + "/" + ds_name + "_edge_labels.txt"):
            print("# \t- edge_labels")
            with open(self.PATH + ds_name + "/" + ds_name + "_edge_labels.txt", "r") as f:
                edge_labels = [int(i) for i in list(f)]
            f.closed

            l_el = []
            for i in xrange(num_graphs + 1):
                g = graph_db[graph_indicator[i]]
                l_el.append(g.new_edge_property("int"))

            for i, l in enumerate(edge_labels):
                g_id = edge_indicator[i]
                g = graph_db[g_id]

                l_el[g_id][edge_list[i]] = l
                g.ep.el = l_el[g_id]

        # Edge Attributes
        if path.exists(self.PATH + ds_name + "/" + ds_name + "_edge_attributes.txt"):
            print("# \t- edge_attributes")
            with open(self.PATH + ds_name + "/" + ds_name + "_edge_attributes.txt", "r") as f:
                edge_attributes = [map(float, i.split(',')) for i in list(f)]
            f.closed

            l_ea = []
            for i in xrange(num_graphs + 1):
                g = graph_db[graph_indicator[i]]
                l_ea.append(g.new_edge_property("vector<float>"))

            for i, l in enumerate(edge_attributes):
                g_id = edge_indicator[i]
                g = graph_db[g_id]

                l_ea[g_id][edge_list[i]] = l
                g.ep.ea = l_ea[g_id]

        return graph_db

    def read_classes(self,ds_name):
        pre = ""
        # Classes
        with open(self.PATH + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
            classes = [int(i) for i in list(f)]
        f.closed

        return classes



    def write_lib_svm(self,gram_matrix, classes, name):
        with open(name, "w") as f:
            k = 1
            for c, row in zip(classes, gram_matrix):
                s = ""
                s = str(c) + " " + "0:" + str(k) + " "
                for i, r in enumerate(row):
                    s += str(i + 1) + ":" + str(r) + " "
                s += "\n"
                f.write(s)
                k += 1
        f.closed

    def write_gram_matrix(self,gram_matrix, ds_name):
        with open(self.PATH + ds_name + "/" + ds_name + "_gram_matrix_simple", 'w') as f:
            for row in gram_matrix:
                s = ""
                for r in row:
                    s += str(r) + " " 
                s += "\n"
                f.write(s)
        f.closed

    def write_feature_vectors(self,feature_vectors, ds_name, classes=[]):
        
        with open(self.PATH + ds_name + "/" + ds_name + "_feature_vectors", 'w') as f:
            dense = feature_vectors.todense()
            shape = dense.shape
            for i in xrange(shape[0]):
                s = "" if classes == [] else str(classes[i]) + " "
                for j in xrange(shape[1]):
                    s += str(dense[i, j])
                    s += " "
                    
                f.write(s + "\n")
    #         f.write(feature_vectors.todense())
        f.closed

    def write_sparse_gram_matrix(self,gram_matrix,ds_name):
        sps.save_npz(self.PATH + ds_name + "/" + ds_name + '_sparse_gram.npz',gram_matrix.tocoo())

    def read_sparse_gram(self,prefix):
        return sps.load_npz(prefix + "_sparse_gram.npz")

    def read_graph_labels(self,prefix):
        print(prefix)
        graph_labels = []
        with open(prefix + '_graph_labels.txt','r') as f:
            for label in list(f):
                graph_labels.append(int(label.strip()))
        return graph_labels

    def writeToCsv(self,data, prefix, suffix):
        with open(prefix + '_' + suffix + '.csv', 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            for row in data:
                csv_writer.writerow(row)

    def store_lambdas(self,data, prefix, suffix):
        with open(prefix + '_' + suffix + '_lambdas' + '.csv', 'w') as f:
            for entry in data:
                f.write(str(entry) + "\n")

import numpy as np
import os

PATH = os.environ['SEML_DATA'] + "/output/"

def read_gram(ds_name):
    with open(PATH + ds_name + '/' + ds_name + '_gram_matrix_simple','r') as f:
        lines = list(f)
        length = len(lines)
        gram = np.zeros((length,length)) 
           
        for i,row in enumerate(lines):
            for j,col in enumerate(row.strip().split(' ')):
                gram[i][j] = float(col)
    return np.asarray(gram)

def read_graph_labels(ds_name):
    graph_labels = []
    with open(PATH + ds_name+'/'+ds_name + '_graph_labels.txt','r') as f:
        for label in list(f):
            graph_labels.append(int(label.strip()))
    return np.asarray(graph_labels)
   
def read_feature_vectors(boundary):
    features = []
    # Load feature vectors
    with open(os.environ['SEML_DATA']+'/output/all/all_feature_vectors') as f:
        i = 0
        for line in f:
            if i >= boundary:
                features.append([float(col) for col in line.strip().split(' ')])
            if i%10 == 0:
                print("# " + str(i))
            i += 1
    return np.asarray(features)

import numpy as np
from sklearn import svm
from svm.graphdb_reader import read_gram,read_graph_labels
#from datasets import *

ds_name = "all"
def main():
    # read graph labels

    graph_labels = np.asarray(read_graph_labels(ds_name))
    # read simple Gram matrix
    gram = read_gram(ds_name) 
    # Train SVM
    clf = svm.SVC(kernel='precomputed')
    print( clf.fit(gram,graph_labels))
    results = clf.predict(gram)
    print (zip(graph_labels,results))

if __name__ == '__main__':
    main()

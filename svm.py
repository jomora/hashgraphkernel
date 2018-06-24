from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods.logging import format_time, time_it
import numpy as np
import csv
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SVM')
    parser.add_argument('-ds','--dataset', action="store", dest="dataset",
        help="The name of the dataset (must match the directory" \
            " name in which the datset is stored, and the prefix" \
            " of all files in the dataset directors.")
    parser.add_argument('-b', '--base',action="store", dest="base",
        help="The base directory in which the datset directory is located")
    parser.add_argument('-c','--components', action="store",type=int,dest='components', default=3)

    args = parser.parse_args()
    # Fetch params
    ds_path = args.base
    ds_name = args.dataset
    components = args.components

    path = ds_path + "/" + ds_name + "/" + ds_name + "_labels-" + str(components) + ".csv"
    print(path)
    with open(path,"r") as f:
        labels = []
        r = csv.reader(f)
        for line in r:
            labels.append(int(line[0].strip()))
    labels = np.array(labels)
    print(labels.shape)
    
    # Init 
    dp = DatasetParser(ds_path)
    
    #Read Gram matrix
    prefix = ds_path + ds_name + '/' + ds_name
    gram = time_it(dp.read_sparse_gram,prefix)

    dense = gram.todense()
    # X_train, X_test = dense[:3000,:3000], dense[3000:,:3000]
    # Y_train, Y_test = labels[:3000], labels[3000:]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test  = train_test_split(dense[:,:3000],labels,train_size=0.7)
    

    print("Shape X_train: " + str(X_train.shape))
    print("Shape X_test: " + str(X_test.shape))
    print("Shape Y_train: " + str(Y_train.shape))
    print("Shape Y_test: " + str(Y_test.shape))
    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    svm = svm.SVC(verbose=True,C=0.001)
    clf = OneVsRestClassifier(estimator=svm)
    # clf = svm.LinearSVC(kernel='precomputed',verbose=True,multi_class="crammer_singer")
    
    print( clf.fit(X_train,Y_train))
    Y_pred = clf.predict(X_test)

    print 'accuracy score: %0.3f' % accuracy_score(Y_test,Y_pred)


if __name__ == "__main__":
    main()

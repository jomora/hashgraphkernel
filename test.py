numpy as np
from sklearn import svm
from svm.graphdb_reader import read_gram,read_graph_labels
import os
ds_name = "all"

# Load feature vectors
#with open(os.environ['SEML_DATA']+'/output/all/all_feature_vectors') as f:
#    features = np.asarray([ float(col) for line in f for col in line.strip().split(' ') ])

# read graph labels
graph_labels = np.asarray(read_graph_labels(ds_name))
# read simple Gram matrix
gram = read_gram(ds_name) 

# Take training data form Gram matrix
splitted = gram[:12000,:12000]
splitted_labels = graph_labes[:12000]
print( clf.fit(splitted,splitted_labels))
results = clf.predict(gram[12000:,:12000])

import pickle
import time
with open(time.strftime("%y-%m-%d_%H%M") + "_SVC.pkl","wb") as f:
    pickle.dump(clf,f)



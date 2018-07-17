import os
import numpy as np
import argparse
from sklearn.cluster import SpectralClustering,KMeans

def main():

	parser = argparse.ArgumentParser(description='Run kernel PCA')
	parser.add_argument('-ds','--dataset', action="store", dest="dataset",
	help="The name of the dataset (must match the directory" \
	" name in which the datset is stored, and the prefix" \
	" of all files in the dataset directors.")
	parser.add_argument('-b', '--base',action="store", dest="base",
	help="The base directory in which the datset directory is located")
	parser.add_argument('--clusters',action="store",type=int,dest="clusters",default=3)
	args = parser.parse_args()
	# Fetch params
	ds_path = args.base
	ds_name = args.dataset
	prefix = ds_path + ds_name + '/' + ds_name
	clusters = args.clusters
	data = np.load(prefix + "_project_ratios.txt.npy")
        with open(prefix + "_id_project_map.txt") as f:
            id_project_map = dict([[elems for elems in line.strip().split(" ")] for line in f])
        id_project_map = {int(k):v for (k,v) in id_project_map.items()}

	kmeans = KMeans(n_clusters=clusters,n_init=1 )
	labels = kmeans.fit_predict(data)#[subsampled_indices])
        for i,l in enumerate(labels):
            print("Project %d, label: %d (%s)" % (i,l,id_project_map[i]))

	N = labels.shape[0]

	uni  = np.unique(labels)
	class_dist = []
	for i in uni:
		prior = float(np.sum(labels == i))/N
		class_dist.append((i,prior))
		print("Class %d: %f" % (i,prior))
	print(class_dist)



	# with open(prefix + "_labels-" + str(components) + ".csv","w") as f:
	# 	for label in spectral_labels:
	# 		f.write( str(label) + '\n')

if __name__ == "__main__":
	main()

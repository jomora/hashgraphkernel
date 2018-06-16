import os
import numpy as np
import argparse
from sklearn.cluster import SpectralClustering

def main():

	parser = argparse.ArgumentParser(description='Run kernel PCA')
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
	prefix = ds_path + ds_name + '/' + ds_name
	components = args.components

	readCsv = lambda path : np.genfromtxt(path, delimiter=',')
	filePath = prefix + "_kpca-" + str(components) +".csv"
	print("Reading: " + filePath)
	X_kpca = readCsv(filePath)
	print(X_kpca)

	silhouette_avgs = []
	#subsampled_indices = np.random.choice(X_kpca.shape[0],200,replace=False)
	for components in range(2,15):
		print("Number of components: %d" %components)
		spectral_labels = compute_spectral_clusters(X_kpca, components)

		silhouette_avg, sample_silhouette_labels = compute_silhouette_metrics(X_kpca,spectral_labels)
		silhouette_avgs.append(silhouette_avg)

		print("Silhouette Average:")
		print(silhouette_avg)
		print("Silhouette Sample Labels:")
		print(sample_silhouette_labels)
		print

	print("Max silhouette_avg: %f for component %d" % (np.max(silhouette_avgs),np.argmax(silhouette_avgs)+2))
	with open(prefix + "_labels-" + str(components) + ".csv","w") as f:
		for label in spectral_labels:
			f.write( str(label) + '\n')

def compute_silhouette_metrics(X_kpca,spectral_labels):
	# compute silhouette scores
	import sklearn.metrics as skm
	silhouette_avg = skm.silhouette_score(X_kpca,spectral_labels)
	sample_silhouette_labels = skm.silhouette_samples(X_kpca,spectral_labels)
	return silhouette_avg, sample_silhouette_labels

def compute_spectral_clusters(X_kpca,components):
	spectral = SpectralClustering(n_clusters=components,eigen_solver='arpack')#,affinity="nearest_neighbors")
	spectral_labels = spectral.fit_predict(X_kpca)#[subsampled_indices])
	N = spectral_labels.shape[0]
	uni  = np.unique(spectral_labels)
	class_dist = []
	for i in uni:
		prior = float(np.sum(spectral_labels == i))/N
		class_dist.append((i,prior))
		print("Class %d: %f" % (i,prior))
	print(class_dist)
	return spectral_labels

if __name__ == "__main__":
	main()

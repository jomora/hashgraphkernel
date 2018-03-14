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


	#subsampled_indices = np.random.choice(X_kpca.shape[0],200,replace=False)

	spectral = SpectralClustering(n_clusters=3,eigen_solver='arpack')#,affinity="nearest_neighbors")
	spectral_labels = spectral.fit_predict(X_kpca)#[subsampled_indices])
	with open(prefix + "_labels-" + str(components) + ".csv","w") as f:
		for label in spectral_labels:
			f.write( str(label) + '\n')

if __name__ == "__main__":
	main()

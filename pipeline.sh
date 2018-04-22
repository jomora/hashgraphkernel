#!/bin/bash

# README
#
# The script expects three parameters ordered as listed:
# <input_dir>	Input directory: a directory containing *.class files (also in subfolders,
# 	e.g. unpacked JARs)
# <output_dir>	Name for new output directory: the name of the newly created input directory
# 	The directory will be create relative to the provided input directory
# <dataset>	Name for new dataset: a name for the newly created dataset. Output data will
#		be stored in a subdirectory of the output directory named like the dataset
#
# How to run the script:
# ./pipeline.sh <input_dir> <output_dir> <dataset>
# Example:
# ./pipeline.sh $SEML_DATA/examples/test/ examples_out my_dataset
# Hint:
# Although $SEML_DATA is not necessary to run the script, it would be helpful to
# create a dedicated directory and store its path as an environment variable.
# Throughout multiple runs of the script with different datasets and possibly
# different output folders, $SEML_DATA will look as follows:
# $SEML_DATA/
#	- <input_dir_1>/
#		-	<dataset_a>/
#		-	<dataset_b>/
# - <input_dir_2>/
#		-	...
# - <input_dir_n>/
#		-	...
#	- <output_dir_1>/
#		-	<dataset_c>/
#		-	<dataset_d>/
# - <output_dir_2>/
#		-	...
# - <output_dir_m>/
#		-	...


function read_principal_components(){
	echo "Enter number of principal components:"
	read principal_components
	if [ -z "$principal_components" ]; then
		echo "Using default value: 3"
		principal_components=3
	else
		echo "Using $principal_components principal components"
	fi
}

### ### #### ### ###
# RUN GRAPHBUILDER #
### ### #### ### ###

function graphbuilder(){
	cd "$SEML/code/graphbuilder/"
	# Run ram.sh to set JAVA_OPTS environment variable
	source "./ram.sh"
	# Example call:
	# sbt "run -in=../../data/examples/test -out=examples_out -ds=test"
	echo "Running graphbuilder"
	time sbt "run -in=$input_dir -out=$output_dir -ds=$dataset"
}

### ### #### #### ### ###
# RUN HASH GRAPH KERNEL #
### ### #### #### ### ###
function hgk(){
	echo "Press [ENTER] to go on"
	read

	cd "$SEML/code/examples/hashgraphkernel/"
	echo "Running Hash Graph Kernel"
	ls -1
	time python2 hash_graph_kernels.py -ds "$dataset" -b "$SEML_DATA/$output_dir/"
}
### ### ## ### ###
# RUN KERNEL PCA #
### ### ## ### ###
function kernel_pca(){
	read_principal_components
	time python2 kernel_pca.py -ds "$dataset" -b "$SEML_DATA/$output_dir/" -c "$principal_components"
}

### #### ### ### ### ### ###
# RUN SPECTRAL CLUSTERING  #
### #### ### ### ### ### ###
function spectral_clustering() {
	echo "Press [ENTER] to go on"
	read
	read_principal_components
	time python2 spectral_clustering.py -ds "$dataset" -b "$SEML_DATA/$output_dir/" -c $principal_components
}
### #### ###
# RUN SVM  #
### #### ###
function svm() {
	echo "Press [ENTER] to go on"
	read
	time python2 svm.py -ds "$dataset" -b "$SEML_DATA/$output_dir/"
}



input_dir=$1
output_dir=$2
dataset=$3

echo "Please select what you want to do:"
echo "[1] Graphbuilder"
echo "[2] Hash Graph Kernel"
echo "[3] Kernel PCA"
echo "[4] Spectral Clustering"
echo "[5] SVM"
echo "[6] all of the above"
read selection
case $selection in
	[1]*)
		graphbuilder
		;;

	[2]*)
		hgk
		;;
	[3]*)
		kernel_pca
		;;
	[4]*)
		spectral_clustering
		;;
	[5]*)
		svm
		;;
	[6]*)
		graphbuilder
		hgk
		kernel_pca
		spectral_clustering
		svm
		;;
esac
exit

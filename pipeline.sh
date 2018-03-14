#!/bin/bash

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

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

function graphbuilder_create_db(){
	cd "$SEML_CODE/graphbuilder/"
	# Run ram.sh to set JAVA_OPTS environment variable

	source "./ram.sh" 6

	# Example call:
	# sbt "run -in=../../data/examples/test -out=examples_out -ds=test"
	echo "Running graphbuilder: create-db"
	time $SBT_HOME/bin/sbt "run create-db -in="$SEML_DATA/$input_dir" -out=$output_dir -ds=$dataset"
}

function graphbuilder_read_db(){
	cd "$SEML_CODE/graphbuilder/"
	# Run ram.sh to set JAVA_OPTS environment variable

	source "./ram.sh" 6

	# Example call:
	# sbt "run -in=../../data/examples/test -out=examples_out -ds=test"
	echo "Running graphbuilder: read-db"
	time sbt "run read-db -in="$SEML_DATA/$input_dir" -out=$output_dir -ds=$dataset"
}

### ### #### #### ### ###
# RUN HASH GRAPH KERNEL #
### ### #### #### ### ###
function hgk(){
	echo "Press [ENTER] to go on"
	read

	cd "$SEML_CODE/hashgraphkernel/"
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

	echo "Enter number of clusters to compute:"
	local clusters=0
	read clusters
	if [ -z "$clusters" ]; then
		echo "Using default value: 3"
		clusters=3
	else
		echo "Using $clusters clusters"
	fi

	time python2 spectral_clustering.py -ds "$dataset" -b "$SEML_DATA/$output_dir/" -c $principal_components --clusters $clusters
}
### #### ###
# RUN SVM  #
### #### ###
function svm() {
	echo "Press [ENTER] to go on"
	read
	read_principal_components
	time python2 svm.py -ds "$dataset" -b "$SEML_DATA/$output_dir/" -c "$principal_components"
}

### #### ###
# COMPUTE ...  #
### #### ###
function method_class_frequencies() {
	echo "Press [ENTER] to go on"
	read
	read_principal_components
	time python2 method_class_frequencies.py -in "$SEML_DATA/$input_dir" -ds "$dataset" -b "$SEML_DATA/$output_dir/" -c "$principal_components"
}

### #### ### ### ### ### ###
# COMPUTE PROJECT CLUSTERS  #
### #### ### ### ### ### ###
function project_clustering() {
	echo "Press [ENTER] to go on"
	echo "Enter number of clusters to compute:"
	local clusters=0
	read clusters
	if [ -z "$clusters" ]; then
		echo "Using default value: 3"
		clusters=3
	else
		echo "Using $clusters clusters"
	fi

	time python2 project_clustering.py -ds "$dataset" -b "$SEML_DATA/$output_dir/" --clusters $clusters
}

input_dir=$1
output_dir=$2
dataset=$3
SERVER_ENV="./server-env.cfg"
LAPTOP_ENV="./laptop-env.cfg"
if  [ -f "$SERVER_ENV" ] && [ -f "$LAPTOP_ENV" ]; then
    echo "Choose config:"
    echo "[1] $SERVER_ENV"
    echo "[2] $LAPTOP_ENV"
    read -p "Enter choice: " choice
    if [ "$choice" == 1 ] ; then
        echo "Sourcing $SERVER_ENV"
        source "$SERVER_ENV"
    elif [ "$choice" == 2 ] ; then
        echo "Sourcing $LAPTOP_ENV"
        source "$LAPTOP_ENV"
    else
        echo "No proper choice made: exiting"
	exit 0
    fi
elif [ -f "$SERVER_ENV" ] ; then
    echo "Sourcing $SERVER_ENV"
    source "$SERVER_ENV"
elif [ -f "$LAPTOP_ENV" ] ; then
    echo "Sourcing $LAPTOP_ENV"
    source "$LAPTOP_ENV"
else
    echo "Couldn't load environment: $SERVER_ENV/$LAPTOP_ENV not present!"
    exit 1
fi

#${SEML_CODE:?Variable SEML_CODE not set}
#${SEML_DATA:?Variable SEML_DATA not set}

#${SBT_HOME:?Variable SBT_HOME not set}
#${JAVA_HOME:?Variable JAVA_HOME not set}

echo "Please select what you want to do:"
echo "[1] Graphbuilder create-db"
echo "[2] Graphbuilder read-db"
echo "[3] Hash Graph Kernel"
echo "[4] Kernel PCA"
echo "[5] Spectral Clustering"
echo "[6] SVM"
echo "[7] Frequencies"
echo "[8] Project Clustering"
echo "[9] all of the above"
read selection
case $selection in
	[1]*)
		graphbuilder_create_db
		;;
	[2]*)
		graphbuilder_read_db
		;;
	[3]*)
		hgk
		;;
	[4]*)
		kernel_pca
		;;
	[5]*)
		spectral_clustering
		;;
	[6]*)
		svm
		;;
	[7]*)
		method_class_frequencies
		;;
	[8]*)
		project_clustering
		;;
	[9]*)
		graphbuilder_create_db
		graphbuilder_read_db
		hgk
		kernel_pca
		spectral_clustering
		svm
		;;
esac
exit

#!/bin/bash
#
# This script merges the results of the kernel PCA, the id_method map and the method project map
# into one CSV file. 
# Commas in parameter lists are replaced by semicolons. 
# All whitespaces are replaced by commas.
# In addition, a header is prepended based on the number of principal components
basedir=$1
dataset=$2
princ_comps=$3

if [ -z $princ_comps ];then
	echo "Please provide the number of principal components as 3rd parameter"
fi

echo "Basedir: $basedir"
echo "Dataset: $dataset"

# Select necessary files
cd "$basedir/$dataset/"
id_method_map=$(ls | grep "id_method_map\.txt")
echo "$id_method_map"
kpca=$(ls | grep "_kpca-[0-9]*\.csv")
echo "$kpca"
method_project_map=$(ls | grep "_method_project_map\.txt")
echo "$method_project_map"

echo `pwd`
new_dir=mossel
mkdir "$new_dir"
ls .
cp "$id_method_map" "$new_dir/"
cp "$kpca" "$new_dir/" 
cp "$method_project_map" "$new_dir/"

cd "$new_dir"
ls .
echo `pwd`

# Prepare header and echo into dataset.csv
counter=1
header=""
while [ "$counter" -le "$princ_comps" ]; do
	header+="pc$counter"
	if [ $counter -ne $princ_comps ]; then
		header+=","
	fi
	let counter=counter+1
done
header="$header,return,method,project"
echo "$header" > dataset.tmp

cat filter-5-1_id_method_map.txt  | cut -d' ' -f2,3 > id_method_map.tmp
cat filter-5-1_method_project_map.txt | cut -d' ' -f2 > method_project_map.tmp
# Replace comma in method parameter list by semicolon
pr -mts' ' id_method_map.tmp method_project_map.tmp | tr "," ";" | tr " " "," > id_method_project_map.tmp

# Append to dataset.csv
pr -mts',' "$kpca" id_method_project_map.tmp >> dataset.tmp

tr -d "\015" < dataset.tmp > dataset.csv

mv dataset.csv ../
cd ..
rm -r "$new_dir" 


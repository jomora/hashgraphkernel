from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods.logging import format_time, time_it
import numpy as np
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SVM')
    parser.add_argument("-in","--input-dir",action="store", dest="input_dir")
    parser.add_argument('-ds','--dataset', action="store", dest="dataset",
        help="The name of the dataset (must match the directory" \
            " name in which the datset is stored, and the prefix" \
            " of all files in the dataset directors.")
    parser.add_argument('-b', '--base',action="store", dest="base",
        help="The base directory in which the datset directory is located")
    parser.add_argument('-c','--components', action="store",type=int,dest='components', default=3)

    args = parser.parse_args()
    # Fetch params
    input_dir = args.input_dir
    ds_path = args.base
    ds_name = args.dataset
    components = args.components

    root, dirs, files = next(os.walk(input_dir))
    print(root)
    print(dirs)
    print(files)
    print
    base_path = ds_path + "/" + ds_name + "/" + ds_name
    path = ds_path + "/" + ds_name + "/" + ds_name + "_labels-" + str(components) + ".csv"


    with open(base_path + "_labels-" + str(components) + ".csv","r") as f:
      labels = [int(line.strip()) for line in f]

    with open(base_path + "_subgraph_attributes.txt","r") as f:
      subs = [line.strip() for line in f]
    with open(base_path + "_subgraph_project_map.txt","r") as f:
      subgraph_project_map = {int(splitted[0]):splitted[1] for splitted in [line.strip().split(" ") for line in f]}

    # print(subgraph_project_map)

    overall_counts = np.bincount(labels)
    overall_frequencies = overall_counts / float(np.sum(overall_counts))
    columns = np.size(overall_counts)
    # print(overall_counts)
    # print
    # print(overall_frequencies)
    # print
    # print(columns)

    subgraph_label_map = {i:label for i,label in enumerate(labels,1)}
    project_index_map = {path:i for i,path in enumerate(dirs)}
    index_project_map = {i:path for path, i in project_index_map.items()}
    # print(index_project_map)
    print(index_project_map[0])
    # Create matrix of class counts
    project_class_counts = np.zeros((len(dirs),columns))
    for sub_id,label in subgraph_label_map.items():
        row_index = project_index_map[subgraph_project_map[sub_id]]
        project_class_counts[row_index,label] += 1

    print(project_class_counts)
    for project,index in project_index_map.items():
        print( np.array2string(project_class_counts[index]) + " // " + project)

    # For all projects: compute class ratios
    class_ratios = [project_class_counts[i,:] /np.sum(project_class_counts[i,:]) for i in range(project_class_counts.shape[0])]
    print("For all projects: compute class ratios:")
    print(class_ratios)

    # For all classes: compute project ratios
    project_ratios = [project_class_counts[:,j] /np.sum(project_class_counts[:,j]) for j in range(project_class_counts.shape[1])]
    print("For all classes: compute project ratios:")
    print(project_ratios)
    
if __name__ == "__main__":
    main()

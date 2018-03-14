print(__doc__)

# Authors: Mathieu Blondel
#          Andreas Mueller
# License: BSD 3 clause
#
# Modified by: Jonas Molina Ramirez

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods.dataset_parsers import DatasetParser
from auxiliarymethods.logging import format_time, time_it
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import os

import argparse


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

    # Init 
    dp = DatasetParser(ds_path)
    
    #Read Gram matrix
    gram = time_it(dp.read_sparse_gram,prefix)

    # Run kernel PCA
    computeComponents(components,gram.todense(),dp,prefix)


def computeComponents(components,gram,dp,prefix):
    kpca = KernelPCA(n_components=components,kernel="precomputed",fit_inverse_transform=False,gamma=10)
    X_kpca = time_it(kpca.fit_transform,gram)
    time_it(dp.writeToCsv,X_kpca,prefix, 'kpca-'+str(components))
    time_it(dp.store_lambdas,kpca.lambdas_,prefix,'kpca-'+str(components))

if __name__ == "__main__":
    time_it(main)

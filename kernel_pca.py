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

#ds_path = os.environ['SEML_DATA'] + '/output/'
ds_path = "datasets/"
#ds_name = "filter-5"
ds_name = "ENZYMES"
prefix = ds_path + ds_name + '/' + ds_name

def main():
    components = 30
    dp = DatasetParser(ds_path)
    gram = time_it(dp.read_sparse_gram,prefix)
    computeComponents(components,gram.todense(),dp)


def computeComponents(components,gram,dp):
    kpca = KernelPCA(n_components=components,kernel="precomputed",fit_inverse_transform=False,gamma=10)
    X_kpca = time_it(kpca.fit_transform,gram)
    time_it(dp.writeToCsv,X_kpca,prefix, 'kpca-'+str(components))
    time_it(dp.store_lambdas,kpca.lambdas_,prefix,'kpca-'+str(components))

if __name__ == "__main__":
    time_it(main)
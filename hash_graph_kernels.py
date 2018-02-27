
# Copyright (c) 2017 by Christopher Morris
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
import numpy as np
import time
import datetime
from auxiliarymethods.logging import format_time, time_it
from graphkernel import hash_graph_kernel_parallel as rbk_parallel
# The Good
algorithms = "algorithms"
batch_events = "batch-events-1.0.0.RELEASE"
batch_job = 'batch-job-1.0.0.RELEASE'
commons_io = "commons-io-20030203.000550"
commons_lang = 'commons-lang-20030203.000129'
commons_lang3 = 'commons-lang3-3.7'
guava_235 = 'guava-23.5-jre'
junit_412 = 'junit-4.12'
log4j_1217 = 'log4j-1.2.17'
logback_classic = 'logback-classic-1.2.3'
mockito_all = 'mockito-all-2.0.2-beta'
partitioned_batch_job = 'partitioned-batch-job-1.0.0.RELEASE'
platform_208 = 'platform-2.0.8.RELEASE'
slf4j_api_180 = 'slf4j-api-1.8.0-beta0'
slf4j_log4j12_180 = 'slf4j-log4j12-1.8.0-beta0'
spring_context = 'spring-context-5.0.1.RELEASE'
spring_context_indexer = "spring-context-indexer-5.0.1.RELEASE"
spring_context_support = 'spring-context-support-5.0.1.RELEASE'
task_events = 'task-events-1.0.0.RELEASE'
taskprocessor = 'taskprocessor-1.0.0.RELEASE'
tasksink = 'tasksink-1.0.0.RELEASE'
timestamp_task = 'timestamp-task-1.0.0.RELEASE'

# The Bad
spring_security_core = 'spring-security-core-5.0.0.RELEASE'
lucene_core = 'lucene-core-7.1.0'
dom4j = 'dom4j-2.1.0'
slf4j_jdk14 = 'slf4j-jdk14-1.8.0-beta0'
plexus_utils = 'plexus-utils-3.1.0'
powermock_module_junit4 = 'powermock-module-junit4-2.0.0-beta.5'
powermock_api_mockito = 'powermock-api-mockito-1.7.3'
netty_all = 'netty-all-5.0.0.Alpha2'
jsoup = "jsoup-1.11.2"
xstream = 'xstream-1.4.10-java7'
groovy_all = 'groovy-all-3.0.0-alpha-1'

good_datasets = [
algorithms,
batch_events,
batch_job,
commons_io,
commons_lang,
commons_lang3,
# guava_235,
junit_412,
log4j_1217,
logback_classic,
mockito_all,
partitioned_batch_job,
# platform_208,
slf4j_api_180,
slf4j_log4j12_180,
spring_context,
spring_context_indexer,
spring_context_support,
task_events,
taskprocessor,
tasksink,
timestamp_task 
    ]

bad_datasets = [
spring_security_core,
# lucene_core ,
dom4j,
slf4j_jdk14,
plexus_utils,
powermock_module_junit4,
powermock_api_mockito,
# netty_all,
jsoup,
xstream
# groovy_all 
]


def main():
    start = time.time()
    print ("# Program started at " + format_time(start))
    #dataset = "all"
    dataset = "ENZYMES"
    print ("# Processing dataset: " + dataset)
    # Load ENZYMES data set
    graph_db, classes = time_it(dp.read_txt,(dataset))

#     graph_db = dp.read_graph_db(dataset)
    
    # Parameters used: 
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_sp = [False, False, 0]

    # Parameters used: 
    # Number of iterations for WL: 3
    # Compute gram matrix: False, 
    # Normalize gram matrix: False
    # Use discrete labels: False
    kernel_parameters_wl = [3, False, False, 0]

    # Use discrete labels, too
    # kernel_parameters_sp = [False, False, 1]
    # kernel_parameters_wl = [3, False, False, 1]


    # Compute gram matrix for HGK-WL
    # 20 is the number of iterations
#     gram_matrix, feature_vectors = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
#                                         scale_attributes=True, lsh_bin_width=1.0, sigma=1.0, use_gram_matrices=True)
#     # Normalize gram matrix
#     gram_matrix = aux.normalize_gram_matrix(gram_matrix)
#    gram_matrix = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 20,
 #                                       scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
    # Normalize gram matrix
  #  gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    # Compute gram matrix for HGK-SP
    # 20 is the number of iterations
    gram_matrix, feature_vectors = time_it(rbk_parallel.hash_graph_kernel_parallel,graph_db, wl.weisfeiler_lehman_subtree_kernel, 
        kernel_parameters_wl, 10, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0, use_gram_matrices=True)
 
    # Normalize gram matrix
    gram_matrix = aux.normalize_gram_matrix(gram_matrix)

    print (feature_vectors.todense())
    print ("Shape of feature vectors: " + str(feature_vectors.todense().shape))

    # Write out LIBSVM matrix
    #dp.write_lib_svm(gram_matrix, classes, "gram_matrix")

    # Write out simple Gram matrix used for clustering
    time_it(dp.write_gram_matrix,gram_matrix, dataset)
    # Write out simple Gram matrix in sparse format
    
    print("Gram matrix in NPZ format")
    import scipy.sparse as sps
    time_it(dp.write_sparse_gram_matrix,gram_matrix.tocoo(),dataset)
    print("Shape of Gram Matrix: " + str(np.shape(gram_matrix)))

    #dp.write_feature_vectors(feature_vectors, dataset, [])
    end = time.time()
    print ("# Program ended at ") + format_time(start)
    print ("# Duration in [s]: ") + str(end - start)




if __name__ == "__main__":
    main()

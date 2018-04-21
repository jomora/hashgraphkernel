# Read id_method_map to dict(int:(string,string))
import os
import csv
import numpy as np


def readIdMap(base,dataset):
    rows = []
    with open(base + dataset + "/" + dataset + '_id_method_map.txt','r') as file:
        reader = csv.reader(file,delimiter=' ')
        for row in reader:
            rows.append(row)

    return {int(a):(b,c) for a,b,c in rows}


def find_methods_which_contain(word):
    return np.asarray([x[0] for x in list(filter(lambda row: word in row[2].lower(),rows))],dtype=np.int64)

def find_methods_with_return_type(word):
    return np.asarray([x[0] for x in list(filter(lambda row: word in row[1].lower(),rows))],dtype=np.int64)

# The provided data must have a column named 'ID'
def create_attribute(data,word,col_name,label,f):
    """
    This function creates a Glue viz attribute ...

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    subset_group
        A new subset_group 

    """
    # Create and add Glue Viz attribute
    setters = np.zeros(len(data['ID']),)
    setters[f(word)-1] = True
    # 'kpca' is the name of the Data object (drag'n'drop into the iPython terminal)
    data[col_name] = setters # UNCOMMENT
    state = data.id[col_name] == 1.0
    return dc.new_subset_group(label,state)

def createSubsetForProject(i):
    dc.new_subset_group(rows[i],kpca.id['project'] == i)

def createSubsetsForAllProject():
    for i in np.unique(mpm['col2']):
        createSubsetForProject(i)

def createSubsetForGraphLength(length):
    dc.new_subset_group('Length ' + str(length),kpca.id['length'] == length)


def readGraphAttributes():
    rows = []
    with open(os.environ['SEML_DATA'] + '/testoutput/filter-5-1/filter-5-1_graph_attributes.txt','r') as file:
        reader = csv.reader(file,delimiter=',')
        for row in reader:
            tmp_row = []
            for elem in row:
                tmp_row.append(int(elem))
            rows.append(tmp_row)
    return rows

# f = lambda data: data == 10


class SubsetCreator:
    """
    This constructor instantiates a SubsetCreator.

    Parameters
    ----------
    kpca : Data
        A Data instance which contains at least a column named "ID". It holds the result of the kernel PCA.
    id_map : 
        A dict instance whose index correpsonds to the "ID" of kpca. The values are tuples of (String,String) containing (returnType,methodName). 
    mpm:
        A Data instance which contains at least the columns "ID" and "project"
    dc:
        DataComponents
    Returns
    -------
    SubsetCreator
        A new SubsetCreator 

    """
    def __init__(self,kpca,id_map,mpm,dc):
        self.kpca = kpca
        self.id_map = id_map
        self.mpm = mpm
        self.dc = dc

    def subsetOfLength(self,f,graphLengths,label):
        lengths = np.zeros(len(self.kpca['ID']),)
        lengths[f(graphLengths[:,1])] = True
        self.kpca[label] = lengths
        state = self.kpca.id[label] == 1.0
        self.dc.new_subset_group(label,state)


####
base = os.environ['SEML_DATA'] + '/testoutput/'
dataset = "filter-5-1"

id_map = readIdMap(base,dataset)

graphLengths = np.asarray(readGraphAttributes())


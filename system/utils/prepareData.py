'''
File contains all the utility functions like loading and cleaning data,
turning it into trees etc
'''
import numpy as np
from nltk.tree import Tree
import re


def loadData(filepath, shuffle=False):
    '''
    Function that splits the incoming treebank into train/val/test
    Also does some cleaning (removes functional labels)

    Inputs:
    filepath: str, stores the path of the treebank
    shuffle: bool, shuffle or not the treebank


    Outputs:
    train, val, test: lists of str with 80% train, 10% val, 10% test
    '''
    with open(filepath,'r') as f:
        lines = f.readlines()

    # We clean the data to remove functional labels in non-terminals
    for i in range(len(lines)):
        lines[i] = lines[i].split('\n')[0]
        lines[i] = re.sub(r'-[A-Z]{1,}_[A-Z]{1,}','',lines[i]) # Example: PP-P_OBJ --> PP
        lines[i] = re.sub(r'-[A-Z]{1,}','',lines[i]) # Example: NP-OBJ --> NP
        # lines[i] = lines[i].replace('SENT','S') # Replace SENT by S (useless)
    # We shuffle the data if desired
    if shuffle:
        np.random.shuffle(lines)

    # We split the data
    N = len(lines)
    train = lines[:int(0.8*N)]
    val = lines[int(0.8*N):int(0.9*N)]
    test = lines[int(0.9*N):]

    trainTrees = dataToTrees(train)
    valTrees = dataToTrees(val)
    testTrees = dataToTrees(test)
    return trainTrees, valTrees, testTrees

def dataToTrees(dataList):
    '''
    Transforms the input data into NLTK tree

    Input:
    data: list of str

    Output:
    trees: list of NLTK trees describing the input data
    '''
    trees = []
    for line in dataList:
        # In each line, the beginning bracket berfore SENT needs to be removed
        # We also lowercase any incoming token in the tree
        tree = Tree.fromstring(line,read_leaf=lambda leaf:leaf.lower(),remove_empty_top_bracketing=True)
        # We need to only have at maximum rules like A-->B,C for the CYK algorithm
        tree.chomsky_normal_form(horzMarkov=2)
        # We need to remove things like A-->B-->terminal, therefore we collapse it
        # into A+B-->terminal
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)

        trees.append(tree)
    return trees

from utils.prepareData import loadData
from utils.pcfg import PCFG
import re
from nltk.tree import Tree

# First we load the data: we get list of trees for train, val and test data
inputFile = 'data/sequoia_corpus'
trainTrees,valTrees,testTrees = loadData(inputFile, shuffle=False)

# Then we need to build our Probabilistic CFG
pcfg = PCFG(trainTrees)
grammar = pcfg.grammar
for i, prod in enumerate(grammar.productions()):
    if len(prod.rhs()) == 1:
        print('Unary',prod)
    elif len(prod.rhs()) == 2:
        print('Binary',prod)
    else:
        print('NOT NOTHING',prod)
        raise ValueError("The grammar is not CNF!")
# for tree in trainTrees:
    # for production in tree.productions():
        # if production.is_lexical():
            # print(production.lhs())
            # print(production.rhs()[0])

# print(productions)
# import numpy as np
# tags = np.random.randint(1,5,100)
#
# # a = {'Token':[Tag,proba]}
#
# a = {(1,'bite'):23,(12,'Sure'):15}
#
# print(a)
# key1 = 1
# key2 = 'bite'
# if (key1,key2) in a:
#     print('OUI')
# else:
#     print('NON')

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

'''
File used to define a probabilistic context free gramar
Based on the training data of the sequoia treebank
'''
import nltk
import time

class PCFG(object):
    '''
    The PCFG has two main features:
    --> self.grammar: a probabilistic context-free grammar whose terminals are part-of-speech tags
        (source: https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html, cells 20-21)
    --> self.lexicon: a probabilistic lexicon, i.e. triples of the form
        (token, part-of-speech tag, probability) such that the sum of the
        probabilities for all triples for a given token sums to 1.
    '''
    def __init__(self, trees):
        '''
        A PCFG is built thanks to a given list of trees
        Here, trainTrees will serve as the training data which will construct our PCFG
        '''
        self.trees = trees
        self.grammar = self.buildGrammar()
        self.lexicon = self.buildLexicon()

        # self.tokens = []
        # self.posTags = []

    def buildGrammar(self):
        print('Starting building Grammar...')
        start = time.time()
        # We need to modify https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html, cells 20/21
        # So that we only keep terminals which are PoS tags
        productions = []

        # This will be used in the buildLexicon function
        self.tokens = set() # Tokens in the vocabulary
        self.posTags = set() # PoS tags linked to these tokens
        for tree in self.trees:
            for production in tree.productions():
                if not production.is_lexical(): # We only rules which have terminals which are PoS --> we don't keep terminals which are tokens
                    productions.append(production)
                else: # We keep rules which have a token as terminal
                    self.tokens.add(production.rhs()[0])
                    self.posTags.add(production.lhs())
        root = nltk.Nonterminal('SENT')
        grammar = nltk.induce_pcfg(root, productions)

        end = time.time()
        print('... Grammar built. Time: {:2.2}s'.format(end-start))
        return grammar

    def buildLexicon(self):
        print('Starting building Lexicon...')
        start = time.time()
        lexicon = {(token,posTag):0 for token in self.tokens for posTag in self.posTags}
        # First we count the number of times a PoS->token appears for each Pos and each token
        for tree in self.trees:
            # Here we can't use tree.pos()
            for production in tree.productions():
                if production.is_lexical():
                    lexicon[production.rhs()[0],production.lhs()] += 1

        # # Then we calculate the probability P(token,PoS)
        for token in self.tokens:
            sum = 0
            for posTag in self.posTags:
                sum += lexicon[token,posTag]
            for posTag in self.posTags:
                lexicon[token,posTag]/=sum

        end = time.time()
        print('... Lexicon built. Time: {:2.2}s'.format(end-start))

        return lexicon

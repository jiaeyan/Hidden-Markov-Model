'''
Created on Aug 24, 2017
@author: jiajieyan
'''
from nltk.corpus import brown
from HMM import HMM

# a toy test with NLTK's brown corpus
data = brown.tagged_sents()
bound = int(round(len(data)*0.95))
train = data[:bound]
test = data[bound:]
     
hmm = HMM()
hmm.train(train)
hmm.test(test)
print(hmm.T[:, 10].sum() == 1) # test if all transitions from one state add to 1 in Transition matrix.
print(hmm.E[:, 10].sum() == 1) # test if all transitions from one state add to 1 in Emission maxtrix.
print(hmm.Pi.sum() == 1)       # test if all transitions from START state add to 1 in Pi maxtrix.
print(hmm.forward(data[10]) == hmm.backward(data[10])) # test if the likelihood of Forward algorithm equals to Backforward.

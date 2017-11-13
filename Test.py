from nltk.corpus import brown
from HMM import HMM
from Document import Document

# a toy test with NLTK's brown corpus

wrap_data = []
for seq in brown.tagged_sents():
    features, labels = [], []
    for w, pos in seq:
        features.append(w)
        labels.append(pos)
    wrap_data.append(Document(features, labels))

bound = int(round(len(data)*0.8))
train = data[:bound]
test = data[bound:]

print(hmm.T[:, 10].sum() == 1) # test if all transitions from one state add to 1 in Transition matrix.
print(hmm.E[:, 10].sum() == 1) # test if all emissions from one state add to 1 in Emission maxtrix.
print(hmm.Pi.sum() == 1)       # test if all transitions from START state add to 1 in Pi maxtrix.

hmm = HMM()
hmm.train(train)
test(test)

def test(data):
        correct_num = 0.0
        token_num = 0.0
        for seq in data[10:20]:
            true = seq.labels
            sen = seq.features
            ob, T = self.checkOb(seq)
            print(sen)
            result = self.classify(seq)
            print(true)
            print(result)
            B = hmm.backward(ob, T)
            print(sum(B[:, 0] * hmm.P * hmm.E[ob[0]]))
            print(hmm.likelihood(seq))
            print()
            token_num += len(sen)
            for i in range(len(sen)):
                if true[i] == result[i]:
                    correct_num += 1
        print ('\nAccuracy: ' + str(correct_num / token_num)) 



print(hmm.forward(data[10]) == hmm.backward(data[10])) # test if the likelihood of Forward algorithm equals to Backforward.

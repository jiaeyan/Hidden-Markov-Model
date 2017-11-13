'''
Created on Oct 21, 2017

@author: svenyan
'''
from numpy import array, dot, zeros, log
from numpy.random import uniform
from nltk.corpus import brown

S = {0:"T", 1:"A", 2:"END"}
O = {"2":0, "4":1, "0":2}
Pi = array([0.5, 0.5]) # initials, start:[to, row1] & end:[from, row2]

T = array([[0.6, 0.4],  # transitions, from each row a state to each column another state
              [0.2, 0.8]])
E = array([[0.1, 0.3, 0.6], # emission, from each row a state to each column a kind of ob
              [0.5, 0.3, 0.2]])
P = [S, O, Pi, T, E]

#         Pi: s1, s2, s3, ...

#         T:    s1, s2, s3, ...
#             s1
#             s2
#             s3
#             END
#            ...
#         E:   s1, s2, s3, ...
#           o1
#           o2
#           o3
#           o4
#           unk
#           ...
class Document():
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

class HMM():

    def train(self, data):
        '''Create parameter matrices and normalize.
           T: Transiton maxtrix
           E: Emission matrix
           P: Prior matrix
        '''
        print('begin train...')
        T, E, P = self.count(data)
        S = array([T[:, col].sum() for col in range(self.N)]) # a vector recording the number of each state 
        self.T = T / S                                        # denominator = s1_unicount + s_type (include end_state)
        self.E = E / (S - self.N - 1 + self.M)
        self.P = P / P.sum()
    
    def formulate(self, data, supervised = True, S_set = set(), O_set = {'<unk>'}):          # data format: [[(ob1, s1), (ob2, s2), ...], [],...]
        print('formulating...')
        for seq in data:            # O_set: observation set; S_set: state set;
            O_set.update(seq.features)
            if supervised:
                S_set.update(seq.labels)
        self.M, self.N = len(O_set), len(S_set)            # the number of observation type and state type
        self.O = {o:i for i, o in enumerate(O_set)}        # a dict to record o and its id
        self.S = {"<END>":len(S_set), len(S_set):"<END>"}  # since all states transit to end state, it should be included, and we want to make sure it is always at the last of the table for computation convenience
        for i, s in enumerate(S_set):                      # a two-way dict to record s and its id
            self.S[i] = s
            self.S[s] = i
        return zeros((self.N+1, self.N)) + 1, zeros((self.M, self.N)) + 1, zeros(self.N) + 1
    
    def count(self, data):
        T, E, P = self.formulate(data)
        for seq in data:
            ob, lb = seq.features, seq.labels
            P[self.S[lb[0]]] += 1                      # start transition
            E[self.O[ob[-1]], self.S[lb[-1]]] += 1     # for the last s, record emission count
            T[self.S["<END>"], self.S[lb[-1]]] += 1    # and end-transition count
            for i in range(len(ob) - 1):
                o1, s1, s2 = ob[i], lb[i], lb[i+1]
                T[self.S[s2], self.S[s1]] += 1
                E[self.O[o1], self.S[s1]] += 1
        return T, E, P
    
    def forward(self, ob, T):
        F = zeros((self.N, T))
        F[:, 0] = self.P * self.E[ob[0]] # initialize all states from Pi
        for t in range(1, T):            # for each t, for each state, sum(all prev-state * transition * ob)
            F[:, t] = [dot(F[:, t-1], self.T[s]) * self.E[ob[t], s] for s in range(self.N)]
        return F                         # return dot(F[:, -1], self.T[-1])
    
    def viterbi(self, ob, T):
        V = zeros((self.N, T))
        B = zeros((self.N, T))
        V[:, 0] = self.P * self.E[ob[0]]
        for t in range(1, T): # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.N):
                V[s, t], B[s, t] = max([(p, s) for s, p in enumerate(V[:, t-1] * self.T[s] * self.E[ob[t], s])])
        return V, B
#         best = max([(p, s) for s, p in enumerate(V[:, -1] * self.T[-1])])[1]
#         return self.backtrace([], best, B, T - 1)
    
    def backtrace(self, path, best, B, t):
        if t == -1: return path
        path.insert(0, self.S[best])
        return self.backtrace(path, int(B[best][t]), B, t-1)
    
    def backward(self, ob, T):
        B = zeros((self.N, T))
        B[:, -1] = self.T[-1] # np.ones(self.T.shape[1])
        for t in range(T - 2, -1, -1):
            B[:, t] = [sum(B[:, t+1] * self.T[:, s][:-1] * self.E[ob[t+1]]) for s in range(self.N)]
        return B              # return sum(B[:, 0] * self.P * self.E[ob[0]])
    
    def test(self, data):
        print("testing...")
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
            B = self.backward(ob, T)
            print(sum(B[:, 0] * self.P * self.E[ob[0]]))
            print(self.likelihood(seq))
            print()
            token_num += len(sen)
            for i in range(len(sen)):
                if true[i] == result[i]:
                    correct_num += 1
        print ('\nAccuracy: ' + str(correct_num / token_num)) 
    
    def checkOb(self, seq):
        '''Handle unk, >> also convert feature to id.'''
        ob = []
        for o in seq.features:
            if o in self.O: ob.append(self.O[o])
            else: ob.append(self.O['<unk>'])
        return ob, len(ob)
    
    def likelihood(self, seq):
        '''Likelihood: compute the observation probability.'''
        ob, T = self.checkOb(seq)
        F = self.forward(ob, T)
        return dot(F[:, -1], self.T[-1])
    
    def data_likelihood(self, data):
        '''Compute the likelihood of the entire data.'''
        return sum([self.likelihood(seq) for seq in data])
    
    def classify(self, seq):
        '''Decoding: predict the state sequence of given observation sequence.'''
        ob, T = self.checkOb(seq)
        V, B = self.viterbi(ob, T)
        best = max([(p, s) for s, p in enumerate(V[:, -1] * self.T[-1])])[1]
        return self.backtrace([], best, B, T - 1)

    def semisupervised_train(self, data, S):
        '''Learning: the Baum-Welch algorithm (forward-backward algorithm).
           First store all necessary expected counts in relative tables in E-step,
           then maximize the relative probabilities in M-step.
           
           The S (state set) must be given besides unlabeled data.
        '''
        T, E, P = self.formulate(data, False, S)
        self.T, self.E, self.P = self.initialize(T, E, P)
        self.tie = 0
        while True:
            likelihood = self.data_likelihood(data)
            exp_T, exp_E, exp_P = self.E_step(data)
            self.T[:], self.E[:], self.P[:] = self.M_step(exp_T, exp_E, exp_P)
            new_lh = self.data_likelihood(data)
            if self.converged(likelihood, new_lh):
                break
            
    def initialize(self, T, E, P):
        '''Initialize parameter tables with uniform distribution probabilities.'''
        for col in range(self.N):
            arrT = uniform(1, 10, self.N)
            arrE = uniform(1, 10, self.M)
            T[:, col] = arrT/arrT.sum()
            E[:, col] = arrE/arrE.sum()
        arrP = uniform(1, 10, self.N)
        P[:] = arrP/arrP.sum()
        return T, E, P # return log(T), log(E), log(Pi)
    
    def E_step(self, data):
        '''Store relative expected counts.
           A: alpha table
           B: beta table
        '''
        exp_T, exp_E, exp_P = zeros(self.T.shape) + 1, zeros(self.E.shape) + 1, zeros(self.P.shape) + 1
        for seq in data:
            ob, T = self.checkOb(seq)
            A, B = self.forward(ob, T), self.backward(ob, T)
            xi_start = [A[s, 0] * B[s, 0] for s in range(self.N)]
            xi_end = [A[s, T-1] * B[s, T-1] for s in range(self.N)]
            exp_T[-1] += xi_end
            exp_P += xi_start
            exp_E[ob[-1]] += xi_end
            
            for t in range(T-1):
                for j in self.N:     # next state
                    for i in self.N: # current state
                        exp_T[j, i] += A[i, t] * self.T[j, i] * self.E[ob[t+1], j] * B[j][t+1]
                        exp_E[ob[t], i] += A[i, t] * B[i, t]
            
            return exp_T, exp_E, exp_P
               
    def M_step(self, T, E, P):
        for col in range(self.N):
            T[:, col] /= T[:, col].sum()
        for col in range(self.N):
            E[:, col] /= E[:, col].sum()
        P /= P.sum()
        return T, E, P # log(T), log(E), log(P) 
    
    def converged(self, old, new):
        '''If the data likelihood stops changing, stop iteration.'''
        if new > old: self.tie = 0
        if new == old: self.tie += 1
        return self.tie == 3  
    
    
data = brown.tagged_sents()
wrap_data = []
for seq in data:
    features, labels = [], []
    for w, pos in seq:
        features.append(w)
        labels.append(pos)
    wrap_data.append(Document(features, labels))
print('done')
bound = int(round(len(wrap_data)*0.8))
train = wrap_data[:bound]
test = wrap_data[bound:]
     
hmm = HMM()
hmm.train(train)
print(hmm.T.shape)
print(hmm.E.shape)
print(hmm.P.shape)
print(hmm.T[:, 67].sum())
print(hmm.E[:, 13].sum())
print(hmm.P.sum())
hmm.test(test)
# print(hmm.Pi.sum())
# print(len(hmm.Pi))
# print(len(train))
#  
# print(hmm.T.shape)
# print(hmm.T[:,-1].sum())
# print(hmm.T[10].sum())
# print(hmm.T[150].sum())
# print(hmm.T[25].sum())
# print(hmm.T[300].sum())
# print(hmm.E[10].sum())
# print(hmm.E[150].sum())
# print(hmm.E[25].sum())
# print(hmm.E[300].sum())
# print()
# print(hmm.T[:, 0].sum())
# print(hmm.T[:, 100].sum())
# print(len(train))

# hmm = HMM(P)
# print(hmm.forward(["0", "2", "4", "0"]))
# print(hmm.viterbi(["0", "2", "4", "0"]))
# print(hmm.backward(["0", "2", "4", "0"]))
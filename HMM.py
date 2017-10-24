'''
Created on Aug 24, 2017
@author: jiajieyan
'''
import numpy as np

class HMM():
    
    def train(self, data):          # data format: [[(ob1, s1), (ob2, s2), ...], [],...]
        '''Get the observation type and state type to decide the shapes of neccesary matrices.'''
        O, S = {'<unk>'}, {'<UNK>'} # O: observation set; S: state set; UNK: handle unseen data
        for seq in data:
            for ob, s in seq:
                O.add(ob)
                S.add(s)
        self.O = {ob:i for i, ob in enumerate(O)} # a dict to record observation and its id
        self.S = {}
        for i, s in enumerate(S):                 # a two-way dict to record state and its id
            self.S[i] = s
            self.S[s] = i
        self.S["END"] = len(S)                    # since all states transit to end state, it should be included
        self.S[len(S)] = "END"
        self.T, self.E, self.Pi = self.count(data, np.zeros((len(S) + 1, (len(S)))) + 1, np.zeros((len(O), len(S))) + 1, np.zeros(len(S) + 1) + 1)
    
    def count(self, data, T, E, Pi):
        '''Fill in the matrices and normalize.
           T: transiton maxtrix
           E: emission matrix
           Pi: initial transition matrix
        '''
        for seq in data:
            Pi[self.S[seq[0][1]]] += 1                     # start transition
            E[self.O[seq[-1][0]]][self.S[seq[-1][1]]] += 1 # for the last s, record emission count
            T[self.S["END"]][self.S[seq[-1][1]]] += 1      # and to-end-transition count
            for i in range(len(seq) - 1):
                ob1, s1, s2 = seq[i][0], seq[i][1], seq[i+1][1]
                T[self.S[s2]][self.S[s1]] += 1
                E[self.O[ob1]][self.S[s1]] += 1
        return self.normalize(data, T, E, Pi)
    
    def normalize(self, data, T, E, Pi):
        '''Convert integers in matrices to probabilities.'''
        S = np.array([E[:, col].sum() - len(E) for col in range(E.shape[1])]) # frequences of all states, need to subtract all added 1 ob 
        T = T / (S + len(T))             # denominator = s1_unicount + s_type (include end_state)
        E = E / (S + len(E))
        Pi = Pi / (len(data) + len(Pi))  # START_count = len(data), also assume <START><END> instance
        return T, E, Pi
        
    def forward(self, ob):
        M = np.zeros((self.T.shape[1], len(ob)))
        M[:, 0] = self.Pi[:-1] * self.E[self.O[ob[0]]] # initialize all states from Pi
        for t in range(1, len(ob)):                    # for each t, for each state, sum(all prev-state * transition * ob)
            M[:, t] = [np.dot(M[:, t - 1], self.T[s]) * self.E[self.O[ob[t]]][s] for s in range(self.T.shape[1])]
        return np.dot(M[:, -1], self.T[-1])
    
    def backward(self, ob):
        M = np.zeros((self.T.shape[1], len(ob)))
        M[:, -1] = self.T[-1]
        for t in range(len(ob) - 2, -1, -1):
            M[:, t] = [sum(M[:, t + 1] * self.T[:, s][:-1] * self.E[self.O[ob[t + 1]]]) for s in range(self.T.shape[1])]
        return sum(M[:, 0] * self.Pi[:-1] * self.E[self.O[ob[0]]])
    
    def viterbi(self, ob):
        M = np.zeros((self.T.shape[1], len(ob)))
        B = np.zeros((self.T.shape[1], len(ob)))       # this is the backtrace matrix, each cell records the best previous state
        M[:, 0] = self.Pi[:-1] * self.E[self.O[ob[0]]]
        for t in range(1, len(ob)):                    # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.T.shape[1]):
                M[s][t], B[s][t] = max([(p, s) for s, p in enumerate(M[:, t - 1] * self.T[s] * self.E[self.O[ob[t]]][s])])
        best = max([(p, s) for s, p in enumerate(M[:, -1] * self.T[-1])])[1]
        return self.backtrace([], best, B, B.shape[1] - 1)
    
    def backtrace(self, path, best, B, t):
        if t == -1: return path
        path.insert(0, self.S[best])
        return self.backtrace(path, int(B[best][t]), B, t - 1)
    
    def test(self, data):
        correct_num = 0.0
        token_num = 0.0
        for seq in data:
            token_num += len(seq)
            true = [s for ob, s in seq]
            ob = self.checkUNK([ob for ob, s in seq])
            predict = self.viterbi(ob)
            for i in range(len(seq)):
                if true[i] == predict[i]: correct_num += 1
        print ('\nAccuracy: ' + str(correct_num / token_num)) 

    def checkUNK(self, ob):
        '''Check OOVs to make them unk.'''
        for i in range(len(ob)):
            if self.O.get(ob[i], False) == False:
                ob[i] = '<unk>'
        return ob
  

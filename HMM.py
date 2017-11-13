
from numpy import array, dot, zeros

class HMM():

    def train(self, data):
        '''Create parameter matrices and normalize.
           T: transiton maxtrix
           E: emission matrix
           P: prior matrix
        '''
        T, E, P = self.count(data)
        S = array([T[:, col].sum() for col in range(self.N)]) # a vector recording the number of each state 
        self.T = T / S                                        # denominator = s1_unicount + s_type (include end_state)
        self.E = E / (S - self.N - 1 + self.M)
        self.P = P / P.sum()
    
    def formulate(self, data, unsupervised = False, S_set = set()):          # data format: [[(ob1, s1), (ob2, s2), ...], [],...]
        O_set = {'<unk>'} # O: observation set; S: state set; UNK: handle unseen data
        for seq in data:
            for o, s in seq:
                O_set.add(o)
                S_set.add(s)
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
            P[self.S[seq[0][1]]] += 1                      # start transition
            E[self.O[seq[-1][0]], self.S[seq[-1][1]]] += 1 # for the last s, record emission count
            T[self.S["<END>"], self.S[seq[-1][1]]] += 1    # and end-transition count
            for i in range(len(seq) - 1):
                o1, s1, s2 = seq[i][0], seq[i][1], seq[i+1][1]
                T[self.S[s2], self.S[s1]] += 1
                E[self.O[o1], self.S[s1]] += 1
        return T, E, P
    
    def checkOb(self, seq):
        '''Handle unk, also convert feature to id.'''
        ob = []
        for o, s in seq:
            if o in self.O: ob.append(self.O[o])
            else: ob.append(self.O['<unk>'])
        return ob, len(ob)
    
    def forward(self, ob, T):
        F = zeros((self.N, T))
        F[:, 0] = self.P * self.E[ob[0]]   # initialize all states from Pi
        for t in range(1, T):              # for each t, for each state, sum(all prev-state * transition * ob)
            F[:, t] = [dot(F[:, t - 1], self.T[s]) * self.E[ob[t], s] for s in range(self.N)]
        return dot(F[:, -1], self.T[-1])
    
    def viterbi(self, ob, T):
        V = zeros((self.N, T))
        B = zeros((self.N, T))
        V[:, 0] = self.P * self.E[ob[0]]
        for t in range(1, T): # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.N):
                V[s, t], B[s, t] = max([(p, s) for s, p in enumerate(V[:, t - 1] * self.T[s] * self.E[ob[t], s])])
        best = max([(p, s) for s, p in enumerate(V[:, -1] * self.T[-1])])[1]
        return self.backtrace([], best, B, T - 1)
    
    def backtrace(self, path, best, B, t):
        if t == -1: return path
        path.insert(0, self.S[best])
        return self.backtrace(path, int(B[best][t]), B, t - 1)
    
    def backward(self, ob, T):
        B = zeros((self.N, T))
        B[:, -1] = self.T[-1]
        for t in range(T - 2, -1, -1):
            B[:, t] = [sum(B[:, t + 1] * self.T[:, s][:-1] * self.E[ob[t + 1]]) for s in range(self.N)]
        return sum(B[:, 0] * self.P * self.E[ob[0]])
    
    def test(self, data):
        correct_num = 0.0
        token_num = 0.0
        for seq in data:
            token_num += len(seq)
            true = [s for ob, s in seq]
            ob = self.checkOb([ob for ob, s in seq])
            predict = self.viterbi(ob)
            for i in range(len(seq)):
                if true[i] == predict[i]: correct_num += 1
        print ('\nAccuracy: ' + str(correct_num / token_num))

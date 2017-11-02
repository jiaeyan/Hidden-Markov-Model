# -*- coding: utf-8 -*-

from itertools import product
import numpy as np

S = {"N":0, "V":1, "END":2, 0:"N", 1:"V", 2:"END"}
O = {"John":0, "likes":1, "apples":2, "oranges":3, 0:"John", 1:"likes", 2:"apples", 3:"oranges"}
Pi = np.array([0.8, 0.2])
T = np.array([[1/3, 1/3],
             [1/3, 1/3],
             [1/3, 1/3]])
E = np.array([[0.25, 0.25],
              [0.25, 0.25],
              [0.25, 0.25],
              [0.25, 0.25]])
obs = [["John", "likes", "apples"], ["John", "likes", "oranges"]]

class EM():
    
    def __init__(self, S, O, T, E, Pi, obs):  # this constructor is for EM algorithm
        self.S = S                            # O: observation set, two-way; S: state set, two-way; UNK: handle unseen data
        self.O = O
        self.T = T                            # transition and emission table
        self.E = E
        self.Pi = Pi                          # prior table
        self.obs = obs                        # given observations
        self.s = [k for k in S if type(k) is str and k != "END"]  # pure state set from S
    
    def likelihood(self, o_seq, s_seq): # compute p(xi, yj)
        likelihood = self.Pi[self.S[s_seq[0]]] * self.T[self.S["END"]][self.S[s_seq[len(s_seq)-1]]]
        for i in range(len(s_seq)-1):
            o1, s1, s2 = o_seq[i], s_seq[i], s_seq[i+1]
            likelihood *= self.T[self.S[s2]][self.S[s1]] * self.E[self.O[o1]][self.S[s1]]
        return likelihood
    
    def expectation(self):                # E-step of EM algorithm
        neo_T = np.zeros(self.T.shape)    # these tables are to update the last iteration parameters
        neo_E = np.zeros(self.E.shape)
        neo_Pi = np.zeros(self.Pi.shape)
        for ob in obs:
            s_combs = [list(comb) for comb in product(self.s, repeat = len(ob))]
            ob_lh = sum([self.likelihood(ob, comb) for comb in s_combs])
            for comb in s_combs:
                posterior = self.likelihood(ob, comb) / ob_lh
                self.count(ob, comb, neo_T, neo_E, neo_Pi, posterior)
        return neo_T, neo_E, neo_Pi
    
    def count(self, ob, s_seq, T, E, Pi, posterior):  # fill the tables by conuting
        Pi[self.S[s_seq[0]]] += posterior
        s_seq.append("END")
        for i in range(len(s_seq)-1):
            o1, s1, s2 = ob[i], s_seq[i], s_seq[i+1]
            T[self.S[s2]][self.S[s1]] += posterior
            E[self.O[o1]][self.S[s1]] += posterior         
    
    def maximization(self, neo_T, neo_E, neo_Pi):     # M-step of EM algorithm
        for col in range(neo_T.shape[1]): neo_T[:, col] = neo_T[:, col]/ neo_T[:, col].sum() 
        for col in range(neo_E.shape[1]): neo_E[:, col] = neo_E[:, col]/ neo_E[:, col].sum()
        neo_Pi = neo_Pi/neo_Pi.sum()
        return neo_T, neo_E, neo_Pi
    
    def converge(self):   # set the condition of convergence
        while True:
            print("Transition: \n", self.T)
            print("Emission: \n", self.E)
            print("Initial: \n", self.Pi)
            self.T, self.E, self.Pi = self.maximization(*self.expectation())
            for ob in self.obs: print(ob, ":", self.viterbi(ob))
            print()
            
    def viterbi(self, ob):          # get most possible state sequence
        M = np.zeros((self.T.shape[1], len(ob)))
        B = np.zeros((self.T.shape[1], len(ob)))
        M[:, 0] = self.Pi * self.E[self.O[ob[0]]]
        for t in range(1, len(ob)): # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.T.shape[1]):
                M[s][t], B[s][t] = max([(p, s) for s, p in enumerate(M[:, t - 1] * self.T[s] * self.E[self.O[ob[t]]][s])])
        best = max([(p, s) for s, p in enumerate(M[:, -1] * self.T[-1])])[1]
        return self.backtrace([], best, B, B.shape[1] - 1)
    
    def backtrace(self, path, best, B, t):
        if t == -1: return path
        path.insert(0, self.S[best])
        return self.backtrace(path, int(B[best][t]), B, t - 1)
    
if __name__ == '__main__':
    em = EM(S, O, T, E, Pi, obs)
    em.converge()
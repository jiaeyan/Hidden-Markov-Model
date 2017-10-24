# Hidden_Markov_Model
An implementation of HMM with Numpy matrices, Viterbi, Forward and Backward algorithms involved.

All used matrices are shaped as below (Initial matrix, Transition matrix, Emission matrix):


         Pi: s1, s2, s3, ..., UNK, END       shape = 1 * (len(S)+1) (this 1 is end state)

         T:    s1, s2, s3, ..., UNK          shape = (len(S)+1) * len(S)
             s1
             s2
             s3
             UNK
             END
            ...
         E:   s1, s2, s3, ..., UNK           shape = len(O) * len(S)
           o1
           o2
           o3
           o4
           unk
           ...

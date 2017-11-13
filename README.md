# Hidden_Markov_Model
An implementation of generic HMM with Numpy matrices, Viterbi, Forward, Backward and EM algorithms involved.

All matrices useed are shaped as below (Initial matrix, Transition matrix, Emission matrix):


         Pi: s1, s2, s3, ...       

         T:    s1, s2, s3, ...
             s1
             s2
             s3
             END
            ...
            
         E:   s1, s2, s3, ...          
           o1
           o2
           o3
           o4
           unk
           ...

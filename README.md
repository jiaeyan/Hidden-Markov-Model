# Hidden_Markov_Model
An implementation of generic HMM with Numpy matrices, Viterbi, Forward, Backward, EM algorithms and Baum-Welch (forward-backward) algorithm involved.

All matrices useed are shaped as below (Initial matrix, Transition matrix, Emission matrix):


         P: s1, s2, s3, ...    --> prior table   

         T:    s1, s2, s3, ... --> transiton table
             s1
             s2
             s3
             END
            ...
            
         E:   s1, s2, s3, ...  --> emission table        
           o1
           o2
           o3
           o4
           unk
           ...

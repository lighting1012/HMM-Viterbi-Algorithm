# HMM Viterbi Algorithm Application
## Data warehouse and data mining project (COMP9318)
</br>
Apply viterbi algorithm to parse Australian address strings in Query_file into (Unitnumber, Streetnumber, Streetname, ..., )
</br>
1.transition matrix: 
</br>transition[i][j] means the probability of state i followed by state j.

2.emission matrix: 
</br>emission[i][j] means the probability of state i emits symbol j.

3.top_k_viterbi() function:
</br>This will return k results with top probability. Each result consist of a list of numbers which refer to a sequence of  the indexes of states.

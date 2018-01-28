# Quora-Question-Pairs

## Approach

Using MaLSTM model (Siamese networks + LSTM with Manhattan metric) to detect semantic similarity between question pairs. 
Training dataset used is a subset of the original Quora Question Pairs Dataset (~363K pairs used) 

## LINKS :
 * Kaggle competition - https://www.kaggle.com/c/quora-question-pairs
 * Complete dataset - https://www.kaggle.com/quora/question-pairs-dataset
 * MaLSTM paper - http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf
 

## RESULTS :

Tried different architectures varying hyperparameter, such as optimize, number of hidden states of LSTM cell, 
activation function of LSTM cell, resulting in validation accuracy of ~77%-81%. Weights for most of the models were saved.

Pre-trained model used in the testing script has 81.2% validation accuracy. 15% data was used as the validation set.


### Instructions for use of testing script :

* Ensure all libraries as mentioned in the **requirements.txt** file are downloaded
* Download Google's pre-trained Word2Vec embedding file and keep the extracted file in the same folder where this repo is cloned. 
Link : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
* The testing script can be run from the terminal as shown below :

                                         python test.py s1 s2

Here, s1 and s2 are the two sentences/questions which are being compared semantically. The output is a binary 1/0 implying the pair being 
duplicate/not duplicate. 

Another example :         
      
      python test.py "Who will be the next captain of Indian cricket team?" "Will Indian cricket team have a captain soon?"

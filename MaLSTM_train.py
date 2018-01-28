### Quora Question-Pairs : Identify duplicate questions by detecting common intent

# Importing standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re, nltk, gensim
import itertools


## Reading the questions (training dataset)
train = pd.read_csv('train.csv')

## EDA
train.describe()
train.describe(include = ['O'])

'''
1. 37% duplicate question-pairs in the training set
2. Not all questions are unique

'''

## Importing required NLTK libraries
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

## Preparing list of questions for pre-processing
q1 = pd.Series(train.question1.tolist()).astype(str)
q2 = pd.Series(train.question2.tolist()).astype(str)


# Finding words with stops in-between
dot_words = []
for row in q1.append(q2):
    for word in row.split():
        if '.' in word and len(word)>2:
            dot_words.append(word)

# Calculating most frequently occuring dot words
import collections
c = collections.Counter(dot_words)
c_sorted = sorted(c, key=c.get, reverse=True)

# Based on the above sorted dictionary, we prepare a keep list
common_dot_words = ['u.s.', 'b.tech', 'm.tech', 'st.', 'e.g.', 'rs.', 'vs.', 'mr.',
                    'dr.', 'u.s', 'i.e.', 'node.js']

## Helper functions
# Pre-process and convert text to a list of words
def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            word = word.lower()
            word = re.sub(r"[^a-zA-Z0-9^.']"," ",word)
            word = re.sub(r"what's", "what is ", word)
            word = re.sub(r"\'ve", " have ", word)
            word = re.sub(r"can't", "cannot ", word)
            word = re.sub(r"n't", " not ", word)
            word = re.sub(r"i'm", "i am ", word)
            word = re.sub(r"\'re", " are ", word)
            word = re.sub(r"\'d", " would ", word)
            word = re.sub(r"\'ll", " will ", word)
            # If the word contains numbers with decimals, this will preserve it
            if bool(re.search(r'\d', word) and re.search(r'\.', word)) and word not in keep_list:
                keep_list.append(word)
            # Preserves certain frequently occuring dot words
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                qs.append(p1)
            else : qs.append(word)
        
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus


def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = True, remove_stopwords = True):
    
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    ''' All stopwords except the 'wh-' words are removed '''
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lem = WordNetLemmatizer()
        corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    
    if stemming == True:
        if stem_type == 'snowball':
            stemmer = SnowballStemmer(language = 'english')
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        else :
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
    
    return corpus


def exponent_neg_manhattan_distance(left, right):
    ''' 
    Purpose : Helper function for the similarity estimate of the LSTMs outputs
    Inputs : Two n-dimensional vectors
    Output : Manhattan distance between the input vectors
    
    '''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# Applying the pre-processing function on the combined text corpus
all_corpus = q1.append(q2)
all_corpus = preprocess(all_corpus, keep_list = common_dot_words, remove_stopwords = False)
'''
Pre-processing all the questions in the training set. While pre-processing, text cleaning and
lemmatization are performed. Stopwords removal is not performed here so as to keep the "wh-"
words which form an integral part of a question's meaning

'''

import pickle
# Dump processed list to file for future re-use
with open('outfile', 'wb') as fp:
    pickle.dump(all_corpus, fp)
# Read back processed list from file at a later time
with open ('outfile', 'rb') as fp:
    all_corpus = pickle.load(fp)


# Separating back to the question-pairs
q1 = all_corpus[0:q1.shape[0]]
q2 = all_corpus[q2.shape[0]::]


# Loading pre-trained word vectors
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary = True)
w2v = dict(zip(word2vec_model.wv.index2word, word2vec_model.wv.syn0))
 
 
'''
Now, the idea is to use a Siamese LSTM network with Manhattan distance as the similarity measure.
So, we have two inputs, q1 and q2, where each word has been converted to a word index.
This word index then, links with word2vec model to have 300-dimensional embeddings at the 
embedding layer. Embedding layers are followed by a shared LSTM layer, whose outputs are used 
to calculate the Manhattan distance/similarity coefficient

'''

# Finding the longest question's length to determine maximum sequence length
q1_len = pd.Series(q1).apply(len)
q2_len = pd.Series(q2).apply(len)
max_seq_length = max(max(q1_len), max(q2_len))

sns.distplot(q1_len.append(q2_len), hist = True, kde = False)
'''
Based on the frequency plot, we can see that there are very few questions with length greater 
than 50. Hence, keeping 50 as the limit
'''
max_seq_length = 50


# Prepare word-to-index mapping
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' acts as a placeholder for the zero vector embedding

qs = pd.DataFrame({'q1': q1, 'q2': q2})
questions_cols = ['q1', 'q2']

# Iterate through the text of both questions of each pair
from tqdm import tqdm
for index, row in tqdm(qs.iterrows()):
    
    for question in questions_cols:
        
        q2n = []  # q2n -> numerical vector representation of each question
        for word in row[question]:
            # Check for stopwords who do not have a word2vec mapping and ignore them
            if word in set(stopwords.words('english')) and word not in word2vec_model.vocab:
                continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                q2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                q2n.append(vocabulary[word])

        # Replace questions with equivalent numerical vector/ word-indices
        qs.set_value(index, question, q2n)
    

# Prepare embedding layer
embedding_dim = 300
embeddings = np.random.randn(len(vocabulary)+1, embedding_dim) # Embedding matrix
embeddings[0] = 0 # This is to ignore the zero padding at the beginning of the sequence

# Build the embedding matrix
for word, index in tqdm(vocabulary.items()):
    if word in word2vec_model.vocab:
        embeddings[index] = w2v[word]

del word2vec_model, w2v
                     
# Train-validation split
X = qs[questions_cols]
y = train['is_duplicate']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15)

# Preparing train and validation feature-sets
X_train = {'left': X_train.q1, 'right': X_train.q2}
X_val = {'left': X_val.q1, 'right': X_val.q2}

# Preparing target labels
y_train = y_train.values
y_val = y_val.values


## Importing required keras libraries
import keras
from keras.preprocessing import sequence
from keras.models import Model,Sequential
from keras.layers import LSTM, Embedding, Input, Merge
from keras.optimizers import Adadelta,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


# Truncate and pad input sequences
for dataset, side in itertools.product([X_train, X_val], ['left', 'right']):
    dataset[side] = sequence.pad_sequences(dataset[side], maxlen=max_seq_length)

# Checking shapes and sizes to ensure no errors occur
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(y_train)


## Build the model
# Model variables
n_hidden = 30
#gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 1

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], 
                            input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden, activation = 'relu')

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), 
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Combine all of the above in a Model
model = Model([left_input, right_input], [malstm_distance])

## Adadelta optimizer, with gradient clipping by norm
#opt = Adadelta(clipnorm=gradient_clipping_norm)

model.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['accuracy'])

# Start training
for i in range(0,3):
    malstm_trained = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=n_epoch,
                            validation_data=([X_val['left'], X_val['right']], y_val))
    model.save_weights("model30_relu_epoch_{}.h5".format(i+1))



# Load saved weights and then compile to make it ready for further training or predictions
model.load_weights("model30_relu_epoch_3.h5")
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

## Making predictions on validation set
preds = model.predict([X_val['left'], X_val['right']])
y_pred = [1 if x > 0.5 else 0 for x in preds] 





























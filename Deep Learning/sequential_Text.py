
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import pandas as pd
import numpy as np
import string, os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load  the data
data = ("../Data/twinkle.txt")
text = open(data, 'r', encoding='utf-8').read()
# Convert into lowercase
corpus = text.lower().split("\n")


t = Tokenizer()
"""
Tokenization is a process of extracting tokens (terms / words) from a corpus. 
Python’s library Keras has inbuilt model for tokenization which can be used to obtain the tokens and their index in the corpus.
"""

# TO Get sequence of token
def obtain_tokens_sequence(corpus):
    """
    :param corpus:  contains the text one want the model to learn about.
    :return:
    """

    t.fit_on_texts(corpus)
    """
    fit_on_texts Updates internal vocabulary based on a list of texts. 
    This method creates the vocabulary index based on word frequency. So if you give it something like, "The cat sat on the mat."
     It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value. 0 is reserved for padding. So lower integer means more frequent word
    """

    total_words = len(t.word_index) + 1
    print("Word Index:", t.word_index)
    """
    word_index: dictionary mapping words (str) to their rank/index (int). Only set after fit_on_texts was called.
    The variable total_words contains the total number of different words that have been used.
    """

    input_sequences = []
    for line in corpus:
        token_list = t.texts_to_sequences([line])[0]
        """
        texts_to_sequences Transforms each text in texts to a sequence of integers. 
        So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary
        The token_list variable contains the sentence as a sequence of tokens
        """
        print("token_list", token_list)
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            """
            the n_gram_sequences creates the n-grams. It starts with the first two words, and then gradually adds words
            """

    return input_sequences, total_words


input_sequences, total_words = obtain_tokens_sequence(corpus)
print("total words:", total_words)


# pad sequences is used to make the lenght equal, because not all sequences have the same length
def create_padded_sequences(input_sequences):
    # to pad all sentences to the maximum length of the sentences, we must first find the longest sentence
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    """
    sequences: List of lists, where each element is a sequence.
    maxlen: Int, maximum length of all sequences.
    padding: String, 'pre' or 'post': pad either before or after each sequence. Paddings adds sequences of 0’s before each line of the
    variable input_sequences so that each line has the same length as the longest line.
    """
    predictors, y = input_sequences[:, :-1], input_sequences[:, -1]
    """
    Spliting x and y to predict the next word of a sequence. 
    it takes all tokens except for the last one as X, and take the last one as  y
    """
    label = ku.to_categorical(y, num_classes=total_words)
    """one-hot encode the y to get a sparse matrix that contains a 1 in the column that corresponds to the token, and 0 eslewhere"""

    return predictors, label, max_sequence_len


predictors, label, max_sequence_len = create_padded_sequences(input_sequences)


def generate_model(max_sequence_len, total_words):
    model = Sequential()

    # The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset
    model.add(Embedding(total_words, 10, input_length=max_sequence_len - 1))
    # Add Hidden Layer 1 - LSTM Layer. pass an LSTM with 250 neuron
    model.add(LSTM(250))
    model.add(Dropout(0.1))
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


model = generate_model(max_sequence_len, total_words)
model.fit(predictors, label, batch_size=12, epochs=100, verbose=0)


def sequential_text(input_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        """ a loop that generates for a given number of iterations the next word"""

        # get token
        token_list = t.texts_to_sequences([input_text])[0]
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        # Predict the class. verbosity mode, 0 or 1
        predicted = model.predict_classes(token_list, verbose=0)

        predict_word = ''

        # Get the corresponding work
        for word, index in t.word_index.items():
            if index == predicted:
                predict_word = word
                break

        input_text = input_text + " " + predict_word

    return input_text.title()

sequential_Predict = sequential_text("How I wonder", 3, model, max_sequence_len)
sequential_Predict_2 = sequential_text("He could not", 4, model, max_sequence_len)

print(sequential_Predict)
print(sequential_Predict_2)



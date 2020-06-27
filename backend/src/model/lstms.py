import numpy as np
import pandas as pd

# filename = '../data/headlines.csv'
# csv = pd.read_csv(filename)
# texts = csv['headline'].values
# labels = csv['residuum']


# process labels


# process texts
# stemming / lemmatization
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import spacy
from nltk.stem.porter import PorterStemmer

nlp = spacy.load('en_core_web_sm')

stemmer = PorterStemmer()
text_en = ['this is a very simple sentence about some dogs living in a blue house with a blue small window looking out',
           'this is the second sentence', 'this is the last sentence for today']
text_de = ['Brussels einigt sich mit Gewerkschaften auf sozialverträglichen Jobabbau',
           'das ist der zweite satz', 'hier hört der text wieder auf']


def lemmatize(text):
    doc = nlp(text)
    output = []
    for t in doc:
        print(f'token: {t.text} - lemma: {t.lemma_} - POS Tag: {t.pos_} - stem: {stemmer.stem(t.text)}')
        output.append(t.lemma_)
    return output


for text in text_en:
    lemmatize(text)
    print('________________________________________________________________')
exit()

# tokenization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_words = 10000
max_length = 75
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre', value=0.0)
sequences = np.array(sequences)
print(sequences.shape)

# model
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model

features = 200
input_1 = Input(shape=(max_length,))
embed_1 = Embedding(input_dim=(max_words - 1), output_dim=features, input_length=max_length)(input_1)
bi_lstm_1 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(embed_1)
bi_lstm_2 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(bi_lstm_1)
bi_lstm_3 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=False))(bi_lstm_2)
softmax_1 = Dense(units=classes, activation='softmax')(bi_lstm_3)

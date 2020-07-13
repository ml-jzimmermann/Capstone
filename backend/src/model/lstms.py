import numpy as np
import pandas as pd
from time_series_analysis.time_series import TimeSeries

filename = '../../data/part1.csv'
csv = pd.read_csv(filename)
texts = csv[['Headline', 'Datum']].values

for text_id in range(len(texts)):
    date = texts[text_id][1]
    if len(date) == 10:
        date = date[3:]
    if len(date) == 9:
        date = date[2:]
    if date == 'Datum':
        date = '00.0000'
    texts[text_id][1] = date

time_series = TimeSeries()
labels = time_series.get_residuums_dates(spread=0.025)
print(texts)
print(labels)

from collections import defaultdict
import matplotlib.pyplot as plt

def plot_data():
    months_dict = defaultdict(int)
    for text in texts:
        months_dict[text[1]] += 1

    print(months_dict)

    keys = months_dict.keys()
    values = months_dict.values()

    plt.bar(keys, values)
    plt.xticks(rotation='vertical')
    plt.show()


def merge_data(texts, labels, sliding_window=0):
    final_texts = []
    final_labels = []
    for text_id in range(len(texts)):
        for label_id in range(len(labels) - sliding_window):
            if texts[text_id][1] == labels[label_id + sliding_window][1][3:]:
                final_texts.append(texts[text_id][0])
                final_labels.append(labels[label_id + sliding_window][0])
    return final_texts, final_labels


texts, labels = merge_data(texts, labels, sliding_window=1)
print(texts)
print(labels)
print(len(texts), len(labels))

exit()


# process labels


# process texts
# stemming / lemmatization
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import spacy
from nltk.stem.porter import PorterStemmer

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('de_core_news_sm')

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


for text in text_de:
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

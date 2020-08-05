import numpy as np
import pandas as pd
import spacy
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from time_series_analysis.time_series import TimeSeries

filename = '../../data/airliner_completed.csv'
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
# print(texts)
# print(labels)

from collections import defaultdict
import matplotlib.pyplot as plt


def plot_data(texts):
    months_dict = defaultdict(int)
    for text in texts:
        months_dict[text[1]] += 1

    print(months_dict)

    keys = months_dict.keys()
    values = months_dict.values()

    plt.bar(keys, values)
    plt.xticks(rotation='vertical')
    plt.show()


def plot_data_length(texts):
    values = [len(t) for t in texts]
    print(max(values))
    keys = range(len(texts))

    plt.bar(keys, values)
    plt.xticks(rotation='horizontal')
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
# print(texts)
# print(labels)
print('data dimension:', len(texts), len(labels))


# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('de_core_news_sm')

stemmer = PorterStemmer()
text_en = ['this is a very simple sentence about some dogs living in a blue house with a blue small window looking out',
           'this is the second sentence', 'this is the last sentence for today']
text_de = ['Brussels einigt sich mit Gewerkschaften auf sozialverträglichen Jobabbau',
           'das ist der zweite satz', 'hier hört der text wieder auf']


def test_token(text):
    doc = nlp(text)
    output = []
    for t in doc:
        print(f'token: {t.text} - lemma: {t.lemma_} - POS Tag: {t.pos_} - stem: {stemmer.stem(t.text)}')
        output.append(t.lemma_)
    return output


def lemmatize(texts):
    lemmatized_texts = []
    for document in list(nlp.pipe(texts, disable=['tagger', 'parser', 'ner'])):
        current_text = []
        for token in document:
            current_text.append(token.lemma_)
        lemmatized_texts.append(current_text)
    return lemmatized_texts















from tensorflow.keras.utils import to_categorical

labels = to_categorical(labels)
labels = np.array(labels)
classes = labels.shape[1]
# print(classes)
print('output dimension:', labels.shape)
# print(labels)

def split_test_data(data, split=0.1, random_seed=42):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    split_item = math.floor(split * len(data))
    print('split at: ', split_item)
    x_test, y_test = data[:split_item, 0], data[:split_item, 1:]
    x_train, y_train = data[split_item:, 0], data[split_item:, 1:]
    return x_train, y_train, x_test, y_test


# model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
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


# print(texts[42])
texts = lemmatize(texts)
# print(texts[42])

# plot_data_length(texts)

# tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 25000
max_length = 21
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre', value=0.0)
sequences = np.array(sequences)
print('input dimension:', sequences.shape)

matrix = np.matrix(sequences)
print('unique tokens:', matrix.max())

# categorize labels
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
from tensorflow.keras.callbacks import TensorBoard

epochs = 10
batch_size = 32
features = 200
units = 16
input_1 = Input(shape=(max_length,))
embed_1 = Embedding(input_dim=(max_words - 1), output_dim=features, input_length=max_length)(input_1)
bi_lstm_1 = Bidirectional(LSTM(units=units, activation='tanh', dropout=0.2, return_sequences=True))(embed_1)
# bi_lstm_2 = Bidirectional(LSTM(units=32, activation='tanh', dropout=0.2, return_sequences=True))(bi_lstm_1)
bi_lstm_3 = Bidirectional(LSTM(units=units, activation='tanh', dropout=0.2, return_sequences=False))(bi_lstm_1)
softmax_1 = Dense(units=classes, activation='softmax')(bi_lstm_3)

model = Model(inputs=input_1, outputs=softmax_1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

#log_dir = f'../../logs/airliner_lstm_{classes}_{epochs}_{features}_{batch_size}_{units}/'
#tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='../../images/model_plot_lstm.png', show_shapes=True, show_layer_names=True)
# callbacks=[tensorboard]

model.fit(x=sequences, y=labels, validation_split=0.1, batch_size=batch_size, epochs=epochs)

exit()
# explain predictions
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix

# calculate confusion matrix
y = [np.argmax(v) for v in y_val]
x = [np.argmax(x) for x in model.predict(x_val)]
confusion = confusion_matrix(y, x)
classification = classification_report(y, x)
print(confusion)
print(classification)

# print confusion matrix
import matplotlib.pyplot as plt
labels = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation', 'neutral']
cm_df = pd.DataFrame(confusion, labels, labels)
sn.set(font_scale=1.1, font='Arial')
ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix")
plt.show()


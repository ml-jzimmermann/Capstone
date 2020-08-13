import ktrain
from ktrain import text
import pandas as pd
import random
import numpy as np
import math
from sklearn.utils import class_weight
import spacy
import matplotlib.pyplot as plt
import seaborn as sn
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import collections


def train_model_multi(four_classes=False, epochs=8):
    if four_classes:
        class_names = ['least', 'less', 'more', 'most']
    else:
        class_names = ['less', 'equal', 'more']

    csv_airliner = '../../data/merged_ktrain.csv'
    csv_google = '../../data/merged_ktrain_google_en.csv'

    data1 = pd.read_csv(csv_airliner).values
    data2 = pd.read_csv(csv_google).values

    texts_de = [element[0] for element in data1]
    labels_de = [element[1:] for element in data1]

    texts_en = [element[0] for element in data2]
    labels_en = [element[1:] for element in data2]

    # preprocess text
    nlp_en = spacy.load('en_core_web_sm')
    stemmer = PorterStemmer()
    stoplist_en = stopwords.words('english')

    nlp_de = spacy.load('de_core_news_sm')
    stoplist_de = stopwords.words('german')

    def test_token(text, nlp):
        doc = nlp(text)
        output = []
        for t in doc:
            print(f'token: {t.text} - lemma: {t.lemma_} - POS Tag: {t.pos_} - stem: {stemmer.stem(t.text)}')
            output.append(t.lemma_)
        return output

    def lemmatize_remove_stop(texts, stoplist, nlp):
        lemmatized_texts = []
        for document in list(nlp.pipe(texts, disable=['tagger', 'parser', 'ner'])):
            current_text = []
            for token in document:
                if token.lemma_ not in stoplist:
                    current_text.append(token.lemma_)

            lemmatized_texts.append(' '.join(current_text))
        return lemmatized_texts

    texts_de = lemmatize_remove_stop(texts_de, stoplist_de, nlp_de)
    texts_en = lemmatize_remove_stop(texts_en, stoplist_en, nlp_en)

    def count_plot_words(texts):
        wordlist = []
        for text in texts:
            wordlist.extend(text.split(' '))
        counter = collections.Counter(wordlist)
        ddict = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
        plt.bar(list(ddict.keys())[:75], list(ddict.values())[:75])
        plt.xticks(rotation=90)
        plt.show()

    data = []
    if four_classes:
        for t, label in zip(texts_de, labels_de):
            data.append([t, label[0], label[1], label[2], label[3]])
        for t, label in zip(texts_en, labels_en):
            data.append([t, label[0], label[1], label[2], label[3]])
    else:
        for t, label in zip(texts_de, labels_de):
            data.append([t, label[0], label[1], label[2]])
        for t, label in zip(texts_en, labels_en):
            data.append([t, label[0], label[1], label[2]])
    # print('data:', data[0])

    learning_rate = 5e-5
    batch_size = 64
    max_length = 12

    def split_test_data(data, split=0.1, random_seed=42):
        np.random.seed(random_seed)
        np.random.shuffle(data)
        split_item = math.floor(split * len(data))
        print('split at: ', split_item)
        x_test, y_test = data[:split_item, 0], data[:split_item, 1:]
        x_train, y_train = data[split_item:, 0], data[split_item:, 1:]
        return x_train, y_train, x_test, y_test

    x_train, y_train, x_val, y_val = split_test_data(np.array(data), split=0.15, random_seed=4242)
    y_train = [[int(e) for e in l] for l in y_train]
    y_val = [[int(e) for e in l] for l in y_val]
    print(len(x_train), len(x_val))
    print(len(y_train), len(y_val))

    # print(y_train[423])

    def generate_balanced_weights(y_train):
        y_labels = [y.argmax() for y in np.array(y_train)]
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
        weight_dict = {}
        for key in range(len(class_weights)):
            weight_dict[key] = class_weights[key]
        return weight_dict

    class_weight_dict = generate_balanced_weights(y_train)

    MODEL = 'bert-base-multilingual-uncased'
    transformer = text.Transformer(MODEL, maxlen=max_length, class_names=class_names)
    train_data = transformer.preprocess_train(x_train, y_train)
    val_data = transformer.preprocess_test(x_val, y_val)

    model = transformer.get_classifier()

    learner = ktrain.get_learner(model, train_data=train_data, val_data=val_data, batch_size=batch_size)

    history = learner.fit_onecycle(learning_rate, epochs=epochs, class_weight=class_weight_dict)
    # predictor = ktrain.get_predictor(learner.model, preproc=transformer)
    confusion = learner.evaluate()

    # print confusion matrix
    cm_df = pd.DataFrame(confusion, class_names, class_names)
    sn.set(font_scale=1.1, font='Arial')
    ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.show()

    return {'history': history.history, 'confusion': confusion}


train_model_multi(four_classes=False)

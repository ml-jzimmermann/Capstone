import collections
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

def load_data(filename,if_google):
    csv = pd.read_csv(filename)
    if if_google:
        return csv[['headline']].values
    return csv[['Headline']].values

def stemming(data):
    stemmer = PorterStemmer()
    new_data = []
    for line in data:
        new_data.append(stemmer.stem(line))

    return new_data

def lemmatize_remove_stop(texts, stoplist, if_google):
    nlp = None
    if if_google:
        nlp = spacy.load('en_core_web_sm')
    else:
        nlp = spacy.load('de_core_news_sm')

    lemmatized_texts = []
    for document in list(nlp.pipe(texts, disable=['tagger', 'parser', 'ner'])):
        current_text = []
        for token in document:
            if token.lemma_ not in stoplist:
                current_text.append(token.lemma_)

        lemmatized_texts.append(' '.join(current_text))
    return lemmatized_texts

def remove_symbols(data):
    symbols = [':', '"', "'", '!', '§', '%', '&', '/', '(', ')', '=', '?', '\\', ',','.',';','<','>', '|','-']
    new_wordList = []
    for word in data:
        if word not in symbols:
            new_wordList.extend(word.split(' '))

    return new_wordList

def plot_barchart_words_withoutstopwords(filename, top=50,if_google=False, without_symbol=False, with_stemming=False):
    text = load_data(filename,if_google)
    texts = [element[0] for element in text]

    if with_stemming:
        texts = stemming(texts)

    # nltk.download('stopwords')
    stoplist = None
    if if_google:
        stoplist = stopwords.words('english')
    else:
        stoplist = stopwords.words('german')
    texts = lemmatize_remove_stop(texts, stoplist, if_google)
    wordlist = []
    for text in texts:
        wordlist.extend(text.split(' '))

    if without_symbol:
        wordlist = remove_symbols(wordlist)


    counter = collections.Counter(wordlist)
    counter = dict(counter.most_common(top))
    plot(counter)



def plot_barchart_words(filename, top=50, if_google=False):
    text = load_data(filename, if_google)
    data = countWords(text, top)
    plot(data)


def countWords(text,top):
    col = collections.Counter()
    for line in text:
        col.update(" ".join(line).split(' '))

    return dict(col.most_common(top))

def plot(data):
    plt.bar(data.keys(), data.values())
    # plt.xlim(0,5)
    plt.xticks(rotation=90)
    plt.show()

# plot_barchart_words('../../data/airliner_completed.csv')
# plot_barchart_words_withoutstopwords('../../data/airliner_completed.csv', if_google=False, without_symbol=True, with_stemming=True)


plot_barchart_words('../../data/merged_ktrain_google_en_four.csv', if_google=True)
plot_barchart_words_withoutstopwords('../../data/merged_ktrain_google_en_four.csv', if_google=True, without_symbol=True, with_stemming=False)
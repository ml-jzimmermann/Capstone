import ktrain
from ktrain import text
import pandas as pd
import random
import numpy as np
import math

csv_file = '../../data/merged_ktrain_google_en.csv'
data = pd.read_csv(csv_file).values
print(len(data))

epochs = 3
learning_rate = 5e-5
batch_size = 512
max_length = 21
max_words = 25000


def split_test_data(data, split=0.1, random_seed=42):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    split_item = math.floor(split * len(data))
    print('split at: ', split_item)
    x_test, y_test = data[:split_item, 0], data[:split_item, 1:]
    x_train, y_train = data[split_item:, 0], data[split_item:, 1:]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_val, y_val = split_test_data(data)
print(len(x_train), len(y_train), len(x_val), len(y_val))

MODEL = 'distilbert-base-uncased'
transformer = text.Transformer(MODEL, maxlen=max_length, class_names=['less', 'equal', 'more'])
train_data = transformer.preprocess_train(x_train, y_train)
val_data = transformer.preprocess_test(x_val, y_val)

model = transformer.get_classifier()
learner = ktrain.get_learner(model, train_data=train_data, val_data=val_data, batch_size=batch_size)

learner.fit_onecycle(learning_rate, epochs)

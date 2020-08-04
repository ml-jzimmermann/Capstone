import ktrain
from ktrain import text
import pandas as pd
import random
import numpy as np
import math

#%%

# csv_file = '../../data/merged_ktrain_google_en.csv'
csv_file = '../../data/merged_ktrain_four.csv'

data = pd.read_csv(csv_file).values
print(len(data))

#%%

epochs = 3
learning_rate = 5e-5
batch_size = 32
max_length = 21
max_words = 25000

#%%

def split_test_data(data, split=0.1, random_seed=42):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    split_item = math.floor(split * len(data))
    print('split at: ', split_item)
    x_test, y_test = data[:split_item, 0], data[:split_item, 1:]
    x_train, y_train = data[split_item:, 0], data[split_item:, 1:]
    return x_train, y_train, x_test, y_test

#%%

x_train, y_train, x_val, y_val = split_test_data(data, split=0.05, random_seed=4242)
print(len(x_train), len(y_train), len(x_val), len(y_val))

#%%

from sklearn.utils import class_weight
def generate_balanced_weights(y_train):
    y_labels = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_labels), y_labels)
    weight_dict = {}
    for key in range(len(class_weights)):
        weight_dict[key] = class_weights[key]
    return weight_dict

class_weight_dict = generate_balanced_weights(y_train)
print(class_weight_dict)

#%%

MODEL ='distilbert-base-multilingual-cased'
transformer = text.Transformer(MODEL, maxlen=max_length, class_names=['least','less','more','most'])
train_data = transformer.preprocess_train(x_train, y_train)
val_data = transformer.preprocess_test(x_val, y_val)

#%%

model = transformer.get_classifier()

#%%

learner = ktrain.get_learner(model, train_data=train_data, val_data=val_data, batch_size=batch_size)

#%%

# learner.lr_find(show_plot=True, max_epochs=2)

#%%

learner.fit_onecycle(5e-5, epochs=1, class_weight=class_weight_dict)

#%%

learner.view_top_losses(n=10, preproc=transformer)

#%%

predictor = ktrain.get_predictor(learner.model, preproc=transformer)

#%%

predictor.explain(x_train[741])

#%%

confusion = learner.evaluate()

#%%

# print confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
labels = ['least','less','more','most']
cm_df = pd.DataFrame(confusion, labels, labels)
sn.set(font_scale=1.1, font='Arial')
ax = sn.heatmap(cm_df, cmap="Blues", annot=True, annot_kws={"size": 11}, cbar=False)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix")
plt.show()

#%%



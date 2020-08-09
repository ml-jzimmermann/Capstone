import pickle
import matplotlib.pyplot as plt

data = None
with open('gridsearch_results_google.save', 'rb') as file:
    data = pickle.load(file)

class_dict = {True: '--', False: '-'}
spread_dict = {0.025: 'red', 0.02: 'blue', 0.015: 'yellow', 0.01: 'green', 0.005: 'magenta'}

for run, results in data.items():
    history = data[run]['history']
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    configuration = data[run]['configuration']
    four_classes = configuration['four_classes']
    spread = configuration['spread']
    plt.plot(acc, class_dict[four_classes], color=spread_dict[spread])
    # plt.plot(val_acc, class_dict[four_classes], color=spread_dict[spread])

plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
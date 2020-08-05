from model.bert import train_model
import pickle
from model.create_dataset import generate_dataset

results = {}
index = 0
for sliding_window in range(1, 7):
    for spread in [0.025, 0.02, 0.015, 0.01, 0.005]:
        for four_classes in [True, False]:
            generate_dataset(four_classes=four_classes, sliding_window=sliding_window, spread=spread)
            result = train_model(four_classes=four_classes)
            configuration = {'four_classes': four_classes, 'spread': spread, 'sliding_window': sliding_window}
            result['configuration'] = configuration
            results[index] = result
            index += 1

with open('gridsearch_results.save', 'wb') as file:
    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

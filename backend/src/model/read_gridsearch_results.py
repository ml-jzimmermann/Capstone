import pickle


with open('gridsearch_results.save', 'rb') as file:
    data_dict = pickle.load(file)
    print(data_dict)


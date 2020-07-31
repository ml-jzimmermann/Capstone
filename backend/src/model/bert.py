import ktrain


csv_file = '../../data/merged_ktrain.csv'

max_length = 21
max_words = 25000

# todo: shuffle input data
(x_train, y_train), (x_test, y_test), preprocessing = ktrain.text.texts_from_csv(train_filepath=csv_file, text_column='headline',
                                                        label_columns=['less', 'equal', 'more'], max_features=max_words, maxlen=max_length,
                                                            sep=',', val_pct=0.05, preprocess_mode='bert')

model = ktrain.text.text_classifier(name='bert', train_data=(x_train, y_train), preproc=preprocessing)

learner = ktrain.get_learner(model=model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=32)

learner.fit_onecycle(lr=2e-2, epochs=4)


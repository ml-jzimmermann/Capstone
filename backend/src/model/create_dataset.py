from time_series_analysis.time_series import TimeSeries
import pandas as pd
from tensorflow.keras.utils import to_categorical


def generate_dataset(four_classes=True, spread=0.01, sliding_window=1):
    input_csv_text = '../../data/airliner_completed.csv'
    csv = pd.read_csv(input_csv_text)
    texts = csv[['Text', 'Datum']].values

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
    time_series.plot_results(time_series.get_residuums(), spread=spread)
    labels = time_series.get_residuums_dates(spread=spread, four_cat=four_classes)

    def merge_data(texts, labels, sliding_window=1):
        final_texts = []
        final_labels = []
        for text_id in range(len(texts)):
            for label_id in range(len(labels) - sliding_window):
                if texts[text_id][1] == labels[label_id + sliding_window][1][3:]:
                    final_texts.append(texts[text_id][0])
                    final_labels.append(labels[label_id + sliding_window][0])
        return final_texts, final_labels

    texts, labels = merge_data(texts, labels, sliding_window=sliding_window)
    labels = to_categorical(labels)

    i = 0
    with open('../../data/auto_generated_gridsearch.csv', 'w') as output_file:
        if four_classes:
            output_file.write('headline,least,less,more,most')
        else:
            output_file.write('headline,less,equal,more')
        output_file.write('\n')
        for text, label in zip(texts, labels):
            i = i + 1
            t = text.replace(',', '').replace('"', '').replace("'", '')
            if four_classes:
                output_file.write(f'{t},{int(label[0])},{int(label[1])},{int(label[2])},{int(label[3])}\n')
            else:
                output_file.write(f'{t},{int(label[0])},{int(label[1])},{int(label[2])}\n')

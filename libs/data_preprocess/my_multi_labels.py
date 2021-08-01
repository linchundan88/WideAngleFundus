import pandas as pd

def get_labels(csv_file, class_index=0):
    list_labels = []
    df = pd.read_csv(csv_file)
    labels = df['labels'].tolist()
    for label in labels:
        single_label = int(label.split('_')[class_index])
        list_labels.append(single_label)

    return list_labels
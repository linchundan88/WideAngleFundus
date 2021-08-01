import os
import pandas as pd
import csv
from libs.data_preprocess.my_compute_digest import calcSha1


filename_csv = os.path.join(os.path.abspath('../..'),
               'datafiles', 'v4', 'test.csv')
filename_csv_dest = os.path.join(os.path.abspath('../..'),
               'datafiles', 'v5', 'test.csv')
os.makedirs(os.path.dirname(filename_csv_dest), exist_ok=True)
df = pd.read_csv(filename_csv)

with open(filename_csv_dest, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    csv_writer.writerow(['images', 'labels'])
    list_sha1 = []
    for _, row in df.iterrows():
        image_file = row['images']
        sha1 = calcSha1(image_file)
        if sha1 in list_sha1:
            continue
        else:
            print(image_file)
            list_sha1.append(sha1)
            csv_writer.writerow([image_file, row['labels']])

print('ok')
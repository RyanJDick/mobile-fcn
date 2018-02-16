import numpy as np
from sklearn.model_selection import train_test_split

data_file_path = '../data/road_seg_full_train_file_list.txt'

with open(data_file_path, 'r') as f:
    file_list = f.readlines()

train_list, test_list = train_test_split(file_list, test_size=0.33, random_state=42)
train_file_path = '../data/road_seg_train_split_file_list.txt'
test_file_path = '../data/road_seg_test_split_file_list.txt'

with open(train_file_path, 'w') as f:
    for item in train_list:
        f.write(item)

with open(test_file_path, 'w') as f:
    for item in test_list:
        f.write(item)

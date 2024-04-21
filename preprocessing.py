import pandas as pd

# original datasets

data_set_0 = pd.read_csv('datasets/part-00000.csv')
data_set_1 = pd.read_csv('datasets/part-00001.csv')
data_set_2 = pd.read_csv('datasets/part-00002.csv')
data_set_3 = pd.read_csv('datasets/part-00003.csv')
data_set_4 = pd.read_csv('datasets/part-00004.csv')
data_set_5 = pd.read_csv('datasets/part-00005.csv')
data_set_6 = pd.read_csv('datasets/part-00006.csv')
data_set_7 = pd.read_csv('datasets/part-00007.csv')
data_set_8 = pd.read_csv('datasets/part-00008.csv')
data_set_9 = pd.read_csv('datasets/part-00009.csv')

data_set = pd.concat([data_set_0, data_set_1, data_set_2, data_set_3,
                      data_set_4, data_set_5, data_set_6,
                      data_set_7, data_set_8, data_set_9])

print(len(data_set))

data_set.to_csv('datasets/dataset-before-preprocessing.csv', index=False)

# labels = ["BenignTraffic", "DDoS-TCP_Flood"]
labels = ["BenignTraffic", "DDoS-SYN_Flood"]
new = data_set[data_set.label.isin(labels)]

new.to_csv('datasets/dataset-preprocessed.csv', index=False)


print(len(new))

# dimentionality reduction

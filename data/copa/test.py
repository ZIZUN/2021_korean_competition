import csv

import pandas as pd
smc_dataset = pd.read_csv('SKT_COPA_Train.tsv', sep='\t').dropna(axis=0)

print(smc_dataset.iloc[0,1:].values)
print(len(smc_dataset))
# print(smc_dataset)
# with open('SKT_BoolQ_Dev.tsv') as f:
#     tr = csv.reader(f, delimiter='\t')
#     for row in tr:
#         print(row)
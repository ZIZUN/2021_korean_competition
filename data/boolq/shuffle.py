import csv
import random
random.seed(518)

all_dataset = []
attribute = None
with open('sum.tsv') as f:
    tr = csv.reader(f, delimiter='\t')
    for row in tr:
        if not row[0] == 'ID':
            all_dataset.append(row)
        else:
            attribute = row

# print(all_dataset)

random.shuffle(all_dataset)
print(all_dataset)

f = open('shuffle_sum.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
wr.writerow(attribute)
for row in all_dataset:
    wr.writerow(row)
f.close()
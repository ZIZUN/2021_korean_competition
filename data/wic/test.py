import csv

import pandas as pd

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

smc_dataset = pd.read_csv('NIKL_SKT_WiC_Train.tsv', sep='\t').dropna(axis=0)
row = smc_dataset.iloc[0,1:].values
print(smc_dataset.iloc[0,1:].values)
print(len(smc_dataset))

sentence_1 = row[1]
sentence_2 = row[2]
label = row[3]
s_1, e_1 = row[4], row[5]
s_2, e_2 = row[6], row[7]


left_1, left_2 = tokenizer.tokenize(sentence_1[:s_1]), tokenizer.tokenize(sentence_2[:s_2])
span_1, span_2 = tokenizer.tokenize(sentence_1[s_1:e_1]), tokenizer.tokenize(sentence_2[s_2:e_2])
right_1, right_2 = tokenizer.tokenize(sentence_1[e_1:]), tokenizer.tokenize(sentence_2[e_2:])
sentence_1 = left_1 + ['<t>'] + span_1 + ['</t>'] + right_1
sentence_2 = left_2 + ['<t>'] + span_2 + ['</t>'] + right_2
text = sentence_1 + ['</s>'] + sentence_2

span_front = list(filter(lambda x: text[x] == '<t>', range(len(text))))
span_end = list(filter(lambda x: text[x] == '</t>', range(len(text))))

print(text[span_front[0]+1:span_end[0]])
print(text[span_front[1]+1:span_end[1]])



print(sentence_1, sentence_2)

# s  ='그의 죽음은 타살로 단정이 되었다.'



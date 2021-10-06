import re
from torch.utils.data import Dataset
import torch
import pandas as pd

from transformers import AutoTokenizer

class LoadDataset_boolq(Dataset):
    def __init__(self, corpus_path, seq_len, model):
        self.seq_len = seq_len

        self.corpus_path = corpus_path

        if model == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                                                   bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                   pad_token='<pad>', mask_token='<mask>')
        elif model == "trinity":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
        elif model == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        self.padding = self.tokenizer.pad_token_id
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id

        self.boolq_dataset = pd.read_csv(corpus_path, sep='\t').dropna(axis=0)

        self.dataset_len = len(self.boolq_dataset)

        self.dataset = []

        # max = 0
        for i in range(self.dataset_len):
            row = self.boolq_dataset.iloc[i, 1:4].values

            context = row[0]
            question = row[1]
            label = row[2]

            context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
            question = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(question))

            text = question + [self.sep] + context

            # if len(text) > max:
            #     max = len(text)
            #     print(max)

            if len(text) <= self.seq_len - 2:
                text = [self.start] + text + [self.sep]

                pad_length = self.seq_len - len(text)

                attention_mask = (len(text) * [1]) + (pad_length * [0])
                text = text + (pad_length * [self.padding])
            else:
                text = text[:self.seq_len - 2]
                text = [self.start] + text + [self.sep]
                attention_mask = len(text) * [1]

            model_input = text
            model_label = int(label)

            self.dataset.append({"input_ids": model_input, 'attention_mask': attention_mask, "labels": model_label})

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        output = self.dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}


class LoadDataset_cola(Dataset):
    def __init__(self, corpus_path, seq_len, model, train):
        self.seq_len = seq_len

        self.corpus_path = corpus_path
        self.train = train

        if model == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                                                   bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                   pad_token='<pad>', mask_token='<mask>')
        elif model == "trinity":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
            self.start = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif model == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
            self.start = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif model == "kcbert":
            self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
            self.start = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif model == 'electra':
            self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
            self.start = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id

        self.padding = self.tokenizer.pad_token_id

        self.cola_dataset = pd.read_csv(corpus_path, sep='\t')

        self.dataset_len = len(self.cola_dataset)

        self.dataset = []

        # max = 0
        for i in range(self.dataset_len):
            row = self.cola_dataset.iloc[i, 1:4].values

            if train == "train" and not(row[1] == '*' or pd.isna(row[1])):
                continue

            context = row[2]
            label = row[0]

            text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))

            if len(text) <= self.seq_len - 2:
                text = [self.start] + text + [self.sep]

                pad_length = self.seq_len - len(text)

                attention_mask = (len(text) * [1]) + (pad_length * [0])

                text = text + (pad_length * [self.padding])
            else:
                text = text[:self.seq_len - 2]
                text = [self.start] + text + [self.sep]
                attention_mask = len(text) * [1]

            model_input = text
            model_label = int(label)

            # print(model_input)

            self.dataset.append({"input_ids": model_input, 'attention_mask': attention_mask, "labels": model_label})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        output = self.dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}




class LoadDataset_copa(Dataset):
    def __init__(self, corpus_path, seq_len, model):
        self.seq_len = seq_len

        self.corpus_path = corpus_path

        if model == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                                                   bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                   pad_token='<pad>', mask_token='<mask>')
        elif model == "trinity":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
        elif model == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        self.padding = self.tokenizer.pad_token_id
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id

        self.copa_dataset = pd.read_csv(corpus_path, sep='\t').dropna(axis=0)

        self.dataset = []

        for i in range(len(self.copa_dataset)):
            row = self.copa_dataset.iloc[i, 1:].values

            premise, input, output_1, output_2, label = \
                row[1], row[0], row[2], row[3], row[4]


            # Full-text-forat   (https://arxiv.org/pdf/2004.14074.pdf)
            if premise == '원인':
                sequence_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input+" 왜냐하면 "+ output_1))
                sequence_2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input + " 왜냐하면 " + output_2))
            else:
                sequence_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input+" 그래서 "+ output_1))
                sequence_2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input + " 그래서 " + output_2))


            # Separated-sentence-format   (https://arxiv.org/pdf/2004.14074.pdf)
            # if premise == '원인':
            #     sequence_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output_1)) \
            #                  + [self.sep] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input))
            #     sequence_2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output_2)) \
            #                  + [self.sep] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input))
            # else:
            #     sequence_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input)) \
            #                  + [self.sep] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output_1))
            #     sequence_2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input)) \
            #                  + [self.sep] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(output_2))

            sequence_1, attention_mask_1 = self.make_sequence(sequence_1)
            sequence_2, attention_mask_2 = self.make_sequence(sequence_2)

            self.dataset.append({"input_ids": [sequence_1, sequence_2],
                                 'attention_mask': [attention_mask_1, attention_mask_2], "labels": int(label) - 1})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        output = self.dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}

    def make_sequence(self, sequence):
        if len(sequence) <= self.seq_len - 2:
            sequence = [self.start] + sequence + [self.sep]

            pad_length = self.seq_len - len(sequence)

            attention_mask = (len(sequence) * [1]) + (pad_length * [0])

            sequence = sequence + (pad_length * [self.padding])
        else:
            sequence = sequence[:self.seq_len - 2]
            sequence = [self.start] + sequence + [self.sep]
            attention_mask = len(sequence) * [1]
        return sequence, attention_mask


class LoadDataset_wic(Dataset):
    def __init__(self, corpus_path, seq_len, model, augment=False):
        self.seq_len = seq_len

        self.corpus_path = corpus_path

        if model == "base":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                                                   bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                                   pad_token='<pad>', mask_token='<mask>')
        elif model == "trinity":
            self.tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
        elif model == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<t>', '</t>']})

        self.padding = self.tokenizer.pad_token_id
        self.start = self.tokenizer.bos_token_id
        self.sep = self.tokenizer.eos_token_id

        self.boolq_dataset = pd.read_csv(corpus_path, sep='\t').dropna(axis=0)

        self.dataset_len = len(self.boolq_dataset)

        self.dataset = []
        for i in range(self.dataset_len): #   2 -> left, right sentence exchange for data augmentation
            row = self.boolq_dataset.iloc[i, 1:].values

            if augment:
                repeat=[0,1]
            else:
                repeat=[0]

            for ppp in repeat:
                if ppp == 0:
                    sentence_1, sentence_2, label, s_1, e_1, s_2, e_2 = \
                        row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                elif ppp == 1:
                    sentence_1, sentence_2, label, s_1, e_1, s_2, e_2 = \
                        row[2], row[1], row[3], row[6], row[7], row[4], row[5]
                else:
                    raise NotImplementedError

                left_1, left_2 = self.tokenizer.tokenize(sentence_1[:s_1]), self.tokenizer.tokenize(sentence_2[:s_2])
                span_1, span_2 = self.tokenizer.tokenize(sentence_1[s_1:e_1]), self.tokenizer.tokenize(sentence_2[s_2:e_2])
                right_1, right_2 = self.tokenizer.tokenize(sentence_1[e_1:]), self.tokenizer.tokenize(sentence_2[e_2:])

                sentence_1 = left_1 + ['<t>'] + span_1 + ['</t>'] + right_1
                sentence_2 = left_2 + ['<t>'] + span_2 + ['</t>'] + right_2

                text = ['</s>'] + sentence_1 + ['</s>'] + sentence_2

                # if i < self.dataset_len: #   left, right sentence exchange for data augmentation
                #     text = ['</s>'] + sentence_1 + ['</s>'] + sentence_2
                # else:
                #     text = ['</s>'] + sentence_2 + ['</s>'] + sentence_1

                span_front = list(filter(lambda x: text[x] == '<t>', range(len(text))))
                span_end = list(filter(lambda x: text[x] == '</t>', range(len(text))))

                text = self.tokenizer.convert_tokens_to_ids(text)

                if len(text) <= self.seq_len - 1:
                    text = text + [self.sep]

                    pad_length = self.seq_len - len(text)

                    attention_mask = (len(text) * [1]) + (pad_length * [0])

                    text = text + (pad_length * [self.padding])
                else:
                    text = text[:self.seq_len - 1]
                    text = text + [self.sep]
                    attention_mask = len(text) * [1]

                span_1, span_2 = [span_front[0] + 1, span_end[0]], [span_front[1] + 1, span_end[1]]

                self.dataset.append({"input_ids": text, 'attention_mask': attention_mask, "labels": int(label),
                                    "span_1": span_1, "span_2": span_2})

    def __len__(self):
        # return self.dataset_len
        return len(self.dataset)

    def __getitem__(self, item):
        output = self.dataset[item]
        return {key: torch.tensor(value) for key, value in output.items()}

    def get_tokenizer_len(self):
        return len(self.tokenizer)
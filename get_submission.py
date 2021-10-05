import pandas as pd
from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaForMultipleChoice
import torch.nn.functional as F
import torch
import json

def boolq_evaluation(best_model_path): # index 1-704
    model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="output/boolq_high_lr_1.2e-5_bsz_10/86_4/",  num_labels=2)
    # best_model_path = "output/boolq_high_lr_1.2e-5_bsz_10/86_4/pytorch_model.bin"
    model = RobertaForSequenceClassification.from_pretrained(best_model_path, config=model_config)


    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    boolq_dataset = pd.read_csv('data/boolq/SKT_BoolQ_Test.tsv', sep='\t')  # .dropna(axis=0)
    seq_len = 300

    dataset_len = len(boolq_dataset)

    padding = tokenizer.pad_token_id
    start = tokenizer.bos_token_id
    sep = tokenizer.eos_token_id

    print(dataset_len)
    dataset = []
    for i in range(dataset_len):
        row = boolq_dataset.iloc[i, 1:3].values

        context = row[0]
        question = row[1]

        context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
        question = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))

        text = question + [sep] + context

        if len(text) <= seq_len - 2:
            text = [start] + text + [sep]

            pad_length = seq_len - len(text)

            attention_mask = (len(text) * [1]) + (pad_length * [0])
            text = text + (pad_length * [padding])
        else:
            text = text[:seq_len - 2]
            text = [start] + text + [sep]
            attention_mask = len(text) * [1]

        dataset.append({"input_ids": torch.tensor(text, dtype=torch.long).unsqueeze(0),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)})
        # print(dataset[i])

    model.to('cuda:3')
    model.eval()

    predict_list = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            data = {key: value.to('cuda:3') for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)

            # print(index+1, int(predict[0]))
            predict_list.append({"idx": index+1, "label": int(predict[0])})
            print('boolq',predict_list[index])

    return predict_list


def cola_evaluation(best_model_path): # index 0-1059
    model_config = RobertaConfig.from_pretrained(
        pretrained_model_name_or_path="output/cola_high_lr_1e-5_bsz_16/74_4/", num_labels=2)
    # best_model_path = "output/cola_high_lr_1e-5_bsz_16/74_4/pytorch_model.bin"
    model = RobertaForSequenceClassification.from_pretrained(best_model_path, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    cola_dataset = pd.read_csv('data/cola/NIKL_CoLA_test.tsv', sep='\t')
    seq_len = 40

    # print(cola_dataset.iloc[4, 1:].values)
    # print(len(cola_dataset))

    padding = tokenizer.pad_token_id
    start = tokenizer.bos_token_id
    sep = tokenizer.eos_token_id

    dataset_len = len(cola_dataset)


    dataset = []
    for i in range(dataset_len):
        context = cola_dataset.iloc[i, 1]

        text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))

        if len(text) <= seq_len - 2:
            text = [start] + text + [sep]

            pad_length = seq_len - len(text)

            attention_mask = (len(text) * [1]) + (pad_length * [0])

            text = text + (pad_length * [padding])
        else:
            text = text[:seq_len - 2]
            text = [start] + text + [sep]
            attention_mask = len(text) * [1]

        dataset.append({"input_ids": torch.tensor(text, dtype=torch.long).unsqueeze(0),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)})


    model.to('cuda:3')
    model.eval()

    predict_list = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            data = {key: value.to('cuda:3') for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)
            predict_list.append({"idx": index, "label": int(predict[0])})
            print('cola',predict_list[index])


    return predict_list


def copa_evaluation(best_model_path): # index 1-500
    model_config = RobertaConfig.from_pretrained(
        pretrained_model_name_or_path="output/copa_high_lr_1e-5_bsz_16/91_2/", num_labels=2)
    # best_model_path = "output/copa_high_lr_1e-5_bsz_16/91_2/pytorch_model.bin"
    model = RobertaForMultipleChoice.from_pretrained(best_model_path, config=model_config)

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    copa_dataset = pd.read_csv('data/copa/SKT_COPA_Test.tsv', sep='\t')
    seq_len = 80

    padding = tokenizer.pad_token_id
    start = tokenizer.bos_token_id
    sep = tokenizer.eos_token_id

    dataset_len = len(copa_dataset)

    dataset = []
    for i in range(dataset_len):
        row = copa_dataset.iloc[i, 1:].values

        premise, input, output_1, output_2 = \
            row[1], row[0], row[2], row[3]

        sequence_1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(premise)) + [sep] + \
                     tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input)) \
                     + [sep] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_1))
        sequence_2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(premise)) + [sep] + \
                     tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input)) \
                     + [sep] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output_2))

        def make_sequence(sequence):
            if len(sequence) <= seq_len - 2:
                sequence = [start] + sequence + [sep]

                pad_length = seq_len - len(sequence)

                attention_mask = (len(sequence) * [1]) + (pad_length * [0])

                sequence = sequence + (pad_length * [padding])
            else:
                sequence = sequence[:seq_len - 2]
                sequence = [start] + sequence + [sep]
                attention_mask = len(sequence) * [1]

            return sequence, attention_mask

        sequence_1, attention_mask_1 = make_sequence(sequence_1)
        sequence_2, attention_mask_2 = make_sequence(sequence_2)

        dataset.append({"input_ids": torch.tensor([sequence_1, sequence_2], dtype=torch.long).unsqueeze(0),
                        'attention_mask': torch.tensor([attention_mask_1, attention_mask_2], dtype=torch.long).unsqueeze(0)})

    model.to('cuda:3')
    model.eval()

    predict_list = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            data = {key: value.to('cuda:3') for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)
            predict_list.append({"idx": index+1, "label": int(predict[0])+1})
            print('copa', predict_list[index])

    return predict_list


def wic_evaluation(best_model_path): # index 1-1246
    from util.model import wic_classifier

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    model = wic_classifier(len(tokenizer)+2)
    # best_model_path = 'output/fintuned.model.2000_91.42367066895368.fintune'
    model.load_state_dict(torch.load(best_model_path))

    seq_len = 400

    wic_dataset = pd.read_csv('data/wic/NIKL_SKT_WiC_Test.tsv', sep='\t')#.dropna(axis=0)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<t>', '</t>']})

    padding = tokenizer.pad_token_id
    sep = tokenizer.eos_token_id

    dataset_len = len(wic_dataset)

    dataset = []
    for i in range(dataset_len):
        row = wic_dataset.iloc[i, 1:].values

        sentence_1, sentence_2, label, s_1, e_1, s_2, e_2 = \
            row[1], row[2], row[3], row[4], row[5], row[6], row[7]

        left_1, left_2 = tokenizer.tokenize(sentence_1[:s_1]), tokenizer.tokenize(sentence_2[:s_2])
        span_1, span_2 = tokenizer.tokenize(sentence_1[s_1:e_1]), tokenizer.tokenize(sentence_2[s_2:e_2])
        right_1, right_2 = tokenizer.tokenize(sentence_1[e_1:]), tokenizer.tokenize(sentence_2[e_2:])

        sentence_1 = left_1 + ['<t>'] + span_1 + ['</t>'] + right_1
        sentence_2 = left_2 + ['<t>'] + span_2 + ['</t>'] + right_2

        text = ['</s>'] + sentence_1 + ['</s>'] + sentence_2


        span_front = list(filter(lambda x: text[x] == '<t>', range(len(text))))
        span_end = list(filter(lambda x: text[x] == '</t>', range(len(text))))

        text = tokenizer.convert_tokens_to_ids(text)

        if len(text) <= seq_len - 1:
            text = text + [sep]

            pad_length = seq_len - len(text)

            attention_mask = (len(text) * [1]) + (pad_length * [0])

            text = text + (pad_length * [padding])
        else:
            text = text[:seq_len - 1]
            text = text + [sep]
            attention_mask = len(text) * [1]

        span_1, span_2 = [span_front[0] + 1, span_end[0]], [span_front[1] + 1, span_end[1]]

        dataset.append({"input_ids": torch.tensor(text, dtype=torch.long).unsqueeze(0),
                        'attention_mask': torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
                             "span_1": torch.tensor(span_1, dtype=torch.long).unsqueeze(0) ,
                        "span_2":  torch.tensor(span_2, dtype=torch.long).unsqueeze(0),
                       })

    model.to('cuda:3')
    model.eval()
    print(len(dataset))

    predict_list = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            data = {key: value.to('cuda:3') for key, value in data.items()}
            logits = model.forward(**data)
            predict = F.softmax(logits, dim=1).argmax(dim=1)
            predict_list.append({"idx": index+1, "label": bool(predict[0])})
            print('wic', predict_list[index])

    print(len(predict_list))

    return predict_list


def result_update(task: str):
    with open('./result.json', 'r', encoding='utf-8') as f:
        result = json.load(f)
    if task == 'boolq':
        task_result = boolq_evaluation()
    elif task == 'cola':
        task_result = cola_evaluation()
    elif task == 'copa':
        task_result = copa_evaluation()
    elif task == 'wic':
        task_result = wic_evaluation()
    else:
        raise NotImplementedError
    result[task] = task_result

    with open('./result.json', 'w', encoding='utf-8') as make_file:
        json.dump(result, make_file, indent="\t")

if __name__ == '__main__':
    task = 'wic'
    best_model_path = 'output/fintuned.model.2000_91.42367066895368.fintune'
    result_update(task, best_model_path)
from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaForMultipleChoice, \
    ElectraConfig, ElectraForSequenceClassification, ElectraForMultipleChoice
import torch.nn.functional as F
import torch
import json
from util.dataset import LoadDataset_cola, LoadDataset_copa, LoadDataset_boolq, LoadDataset_wic
from torch.utils.data import DataLoader
from util.model import wic_classifier
import tqdm
import argparse

def boolq_evaluation_v2(best_model_path, test_data_path = 'data/boolq/SKT_BoolQ_Test.tsv',
                        seq_len=300, batch_size=200, device='cuda:0', num_workers=5, model_name='roberta'):
    if model_name =='roberta':
        model_config = RobertaConfig.from_pretrained(
            pretrained_model_name_or_path=best_model_path, num_labels=2)
        model = RobertaForSequenceClassification.from_pretrained(best_model_path, config=model_config)
    else:
        model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path=best_model_path, num_labels=2)
        model = ElectraForSequenceClassification.from_pretrained(best_model_path, config=model_config)

    test_dataset = LoadDataset_boolq(test_data_path, seq_len=seq_len, model=model_name, train='test')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model.to(device)
    model.eval()

    predict_list = []
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):

            data = {key: value.to(device) for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)
            predict_list += predict.tolist()

    result = []
    for i in range(len(predict_list)):
        result.append({"idx": i+1, "label": int(predict_list[i])})

    # print(result)
    return result



def wic_evaluation_v2(best_model_path, test_data_path='data/wic/NIKL_SKT_WiC_Test.tsv', seq_len=400, batch_size=200, device='cuda:0',num_workers=5, model_name='roberta'):
    if model_name == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        model = wic_classifier(resize_token_embd_len=len(tokenizer) + 2, model_name="roberta")
    elif model_name == 'electra':
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = wic_classifier(resize_token_embd_len=len(tokenizer) + 2, model_name="electra")
    elif model_name == 'electra_tunib':
        tokenizer = AutoTokenizer.from_pretrained("tunib/electra-ko-base")
        model = wic_classifier(resize_token_embd_len=len(tokenizer) + 2, model_name="electra_tunib")

    model.load_state_dict(torch.load(best_model_path))

    test_dataset = LoadDataset_wic(corpus_path=test_data_path, seq_len=seq_len, model=model_name, train='test')

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model.to(device)
    model.eval()
    predict_list = []
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):
            data = {key: value.to(device) for key, value in data.items()}
            logits = model.forward(**data)
            predict = F.softmax(logits, dim=1).argmax(dim=1)
            predict_list += predict.tolist()

    result = []
    for i in range(len(predict_list)):
        result.append({"idx": i+1, "label": bool(predict_list[i])})

    # print(result)
    return result

def copa_evaluation_v2(best_model_path, test_data_path = 'data/copa/SKT_COPA_Test.tsv' ,seq_len=80, batch_size=200, device='cuda:0', num_workers=5, model_name='roberta'):
    if model_name =='roberta':
        model_config = RobertaConfig.from_pretrained(
            pretrained_model_name_or_path=best_model_path, num_labels=2)
        model = RobertaForMultipleChoice.from_pretrained(best_model_path, config=model_config)
    else:
        model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path=best_model_path, num_labels=2)
        model = ElectraForMultipleChoice.from_pretrained(best_model_path, config=model_config)


    test_dataset = LoadDataset_copa(corpus_path=test_data_path, seq_len=seq_len, model=model_name, train='test')

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    model.to(device)
    model.eval()
    predict_list = []
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):
            data = {key: value.to(device) for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)
            predict_list += predict.tolist()

    result = []
    for i in range(len(predict_list)):
        result.append({"idx": i+1, "label": int(predict_list[i])+1})

    # print(result)
    return result

def cola_evaluation_v2(best_model_path, test_data_path='data/cola/NIKL_CoLA_test.tsv' , seq_len=40, batch_size=200, device='cuda:0', num_workers=5, model_name='electra'):
    model_config = ElectraConfig.from_pretrained(
        pretrained_model_name_or_path=best_model_path, num_labels=2)
    model = ElectraForSequenceClassification.from_pretrained(best_model_path, config=model_config)

    test_dataset = LoadDataset_cola(corpus_path=test_data_path, seq_len=seq_len, model=model_name, train='test')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model.to(device)
    model.eval()

    predict_list = []
    with torch.no_grad():
        for index, data in tqdm.tqdm(enumerate(test_data_loader)):

            data = {key: value.to(device) for key, value in data.items()}
            output = model.forward(**data)
            predict = F.softmax(output.logits, dim=1).argmax(dim=1)

            predict_list += predict.tolist()

    result = []
    for i in range(len(predict_list)):
        result.append({"idx": i, "label": int(predict_list[i])})

    # print(result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int, default=200, help="set batchsize for evaluation")
    parser.add_argument("--num_worker", required=True, type=int, default=5, help="set numworker for evaluation")
    parser.add_argument("--boolq_data_path", required=True, type=str, default=None, help="set testdatapath for evaluation") # data/boolq/SKT_BoolQ_Test.tsv
    parser.add_argument("--cola_data_path", required=True, type=str, default=None, help="set testdatapath for evaluation")
    parser.add_argument("--copa_data_path", required=True, type=str, default=None, help="set testdatapath for evaluation")
    parser.add_argument("--wic_data_path", required=True, type=str, default=None, help="set testdatapath for evaluation")
    parser.add_argument("--result_file_name", required=True, type=str, default='result_new.json',
                        help="set resultfilename for evaluation")
    parser.add_argument("--device", required=False, type=str, default='cuda:0',
                        help="set device for evaluation")
    # python get_result.py --batch_size 100 --num_worker 5 --boolq_data_path 'data/boolq/SKT_BoolQ_Test.tsv' --cola_data_path 'data/cola/NIKL_CoLA_test.tsv' --copa_data_path 'data/copa/SKT_COPA_Test.tsv' --wic_data_path 'data/wic/NIKL_SKT_WiC_Test.tsv' --result_file_name 'result.json'

    args = parser.parse_args()

    boolq_seq_len = 512
    cola_seq_len = 200
    copa_seq_len = 200
    wic_seq_len = 512
    device = args.device

    result_json = {}
    result_json['boolq'] = None
    result_json['copa'] = None
    result_json['wic'] = None
    result_json['cola'] = None

    ### boolq ###

    result_list = []
    result_ensemble = []

    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_1_5750_90.96_573_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_2_5900_88.66_611_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_3_1200_86.66_1044_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_4_5150_89.66_1040_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_5_4350_88.5_300_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_6_1690_88.0_675_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_7_1775_91.0_779_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_8_3740_90.33_1179_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_9_1780_89.33_610_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_10_5510_89.66_277_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))
    result_list.append(boolq_evaluation_v2(best_model_path='output/boolq/_11_2795_90.03_628_roberta', test_data_path=args.boolq_data_path,
                                      seq_len=boolq_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='roberta'))


    for i in range(len(result_list[0])):   # init result_ensemble
        result_ensemble.append({"idx": i, "label": 0})

    for result in result_list:  # sum result_ensemble
        for i, id_and_label in enumerate(result):
            result_ensemble[i]['label'] += id_and_label['label']

    # print(result_ensemble)

    for i in range(len(result_list[0])):   # init result_ensemble
        if result_ensemble[i]['label'] >= 6:
            result_ensemble[i]['label'] = 1
        else:
            result_ensemble[i]['label'] = 0

    # print(result_ensemble)

    result_json['boolq'] = result_ensemble

    #############

    ### cola ###

    result_list = []
    result_ensemble = []

    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_1_1050_78.57_1000_electra', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_2_818_79.9_1060_electra_1.6e_-', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_3_912_78.5_1053_electra', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_4_853_78.10_234_electra', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_5_1088_79.0_131_electra', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_6_950_77.2_4121_electra_tunib', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra_tunib'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_7_1380_77.8_4127_electra_tunib', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra_tunib'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_8_1440_78.5_48_electra_tunib', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra_tunib'))
    result_list.append(cola_evaluation_v2(best_model_path='output/cola/_9_1845_79.0_4124_electra_tunib', test_data_path=args.cola_data_path,
                                      seq_len=cola_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker, model_name='electra_tunib'))


    for i in range(len(result_list[0])):   # init result_ensemble
        result_ensemble.append({"idx": i, "label": 0})

    for result in result_list:  # sum result_ensemble
        for i, id_and_label in enumerate(result):
            result_ensemble[i]['label'] += id_and_label['label']

    # print(result_ensemble)

    for i in range(len(result_list[0])):   # init result_ensemble
        if result_ensemble[i]['label'] >= 5:
            result_ensemble[i]['label'] = 1
        else:
            result_ensemble[i]['label'] = 0

    # print(result_ensemble)

    result_json['cola'] = result_ensemble

    #############

    ### copa ###

    result_list = []
    result_ensemble = []

    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_1_500_88.96_201_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_2_1400_89.0_202_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_3_1450_95.6_303_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_4_1550_93.33_304_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_5_1000_91.0_174_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_6_750_91.33_406_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_7_1800_89.66_407_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_8_1900_89.0_508_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))
    result_list.append(copa_evaluation_v2(best_model_path='output/copa/_9_1650_91.33_949_roberta', test_data_path=args.copa_data_path,
                                         seq_len=copa_seq_len, batch_size=args.batch_size, device=device,num_workers=args.num_worker, model_name='roberta'))

    for i in range(len(result_list[0])):   # init result_ensemble
        result_ensemble.append({"idx": i, "label": 0})

    for result in result_list:  # sum result_ensemble
        for i, id_and_label in enumerate(result):
            result_ensemble[i]['label'] += id_and_label['label'] - 1

    # print(result_ensemble)

    for i in range(len(result_list[0])):   # init result_ensemble
        if result_ensemble[i]['label'] >= 5:
            result_ensemble[i]['label'] = 2
        else:
            result_ensemble[i]['label'] = 1

    # print(result_ensemble)

    result_json['copa'] = result_ensemble

    ############

    ### wic ###

    result_list = []
    result_ensemble = []

    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_1_10215_94.48_1440_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_2_4065_94.25_1550_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_3_5085_94.25_1450_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_4_4600_94.75_693_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_5_7050_94.5_10_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_6_7655_94.5_1954_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_7_8050_91.75_999_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_8_2380_93.75_311_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))
    result_list.append(wic_evaluation_v2(best_model_path='output/wic/_9_7875_95.25_90_roberta', test_data_path=args.wic_data_path,
                                         seq_len=wic_seq_len, batch_size=args.batch_size, device=device, num_workers=args.num_worker,
                                         model_name='roberta'))

    for i in range(len(result_list[0])):  # init result_ensemble
        result_ensemble.append({"idx": i, "label": False})

    for result in result_list:  # sum result_ensemble
        for i, id_and_label in enumerate(result):
            result_ensemble[i]['label'] += id_and_label['label']

    # print(result_ensemble)

    for i in range(len(result_list[0])):  # init result_ensemble
        if result_ensemble[i]['label'] >= 5:
            result_ensemble[i]['label'] = True
        else:
            result_ensemble[i]['label'] = False

    # print(result_ensemble)

    result_json['wic'] = result_ensemble

    #############

    with open('result/' + args.result_file_name, 'w', encoding='utf-8') as make_file:
        json.dump(result_json, make_file, indent="\t")

    print('Saved result !!')
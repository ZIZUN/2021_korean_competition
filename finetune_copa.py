import argparse

from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification,GPT2Config, RobertaForSequenceClassification, RobertaConfig, \
    BertForSequenceClassification, BertConfig, ElectraForSequenceClassification, ElectraConfig, RobertaForMultipleChoice, ElectraForMultipleChoice
from util.trainer import Trainer
from util.dataset import LoadDataset_copa

import torch

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset")
parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

parser.add_argument("--model", type=str, required=True, help="model (base,trinity)")
parser.add_argument("--ddp", type=bool, default=False, help="for distrbuted data parrerel")
parser.add_argument("--local_rank", type=int, help="for distrbuted data parrerel")

parser.add_argument("--input_seq_len", required=True, type=int, default=512, help="maximum sequence input len")

parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=1, help="printing loss every n iter: setting n")
parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

parser.add_argument("--accumulate", type=int, default=1, help="accumulation step")
parser.add_argument("--seed", type=int, default=42, help="seed")

args = parser.parse_args()

if args.ddp:
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


print("Loading Train Dataset", args.train_dataset)
train_dataset = LoadDataset_copa(args.train_dataset, seq_len=args.input_seq_len, model=args.model)

print("Loading Test Dataset", args.test_dataset)
test_dataset = LoadDataset_copa(args.test_dataset, seq_len=args.input_seq_len, model=args.model) \
    if args.test_dataset is not None else None

if args.ddp:
    print("Creating Dataloader")
    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
else:
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

if args.ddp:
    print("Creating Dataloader")
    test_sampler = DistributedSampler(test_dataset)
    test_data_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None
else:
    test_data_loader = DataLoader(test_dataset, batch_size=200, num_workers=args.num_workers) \
        if test_dataset is not None else None


if args.model =='base':
    NotImplementedError
    # model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path= "skt/kogpt2-base-v2",    num_labels=2)
    # model = GPT2ForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", config=model_config)
elif args.model =='trinity':
    NotImplementedError
    # model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="skt/ko-gpt-trinity-1.2B-v0.5",    num_labels=2)
    # model = GPT2ForSequenceClassification.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5", config=model_config)
elif args.model == 'roberta':
    model_config = RobertaConfig.from_pretrained(pretrained_model_name_or_path="klue/roberta-large")#, num_labels=2)
    model = RobertaForMultipleChoice.from_pretrained("klue/roberta-large", config=model_config)
elif args.model == 'electra':
    model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path="monologg/koelectra-base-v3-discriminator",
                                                 num_labels=2)
    model = ElectraForMultipleChoice.from_pretrained("monologg/koelectra-base-v3-discriminator", config=model_config)
elif args.model == 'electra_tunib':
    model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path="tunib/electra-ko-base",
                                                 num_labels=2)
    model = ElectraForMultipleChoice.from_pretrained("tunib/electra-ko-base", config=model_config)
elif args.model == 'electra_kor':
    model_config = ElectraConfig.from_pretrained(pretrained_model_name_or_path="kykim/electra-kor-base",
                                                 num_labels=2)
    model = ElectraForMultipleChoice.from_pretrained("kykim/electra-kor-base", config=model_config)

print("Creating Trainer")
trainer = Trainer(task='copa', model=model, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                      with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                  distributed = args.ddp, local_rank = args.local_rank, accum_iter= args.accumulate, seed= args.seed, model_name=args.model)

print("Training Start")
for epoch in range(args.epochs):
    if args.ddp:
        train_sampler.set_epoch(epoch)
        trainer.train(epoch)

        # if args.local_rank == 0:
        #     trainer.save(epoch, args.output_path)
    else:
        trainer.train(epoch)
        # trainer.save(epoch, args.output_path)
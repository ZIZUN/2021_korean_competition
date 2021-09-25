##   ex) bash cola.sh base 4
##   ex) bash cola.sh base 4 ddp 4
model=$1
bsz=$2
ddp=$3
ngpu_ddp=$4

# (DDP) or (nn.dataparallel, cpu)
if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd}python -m torch.distributed.launch --nproc_per_node=${ngpu_ddp} --master_port=75128"
else
    cmd="${cmd}python"
fi

cmd="${cmd} finetune_wic.py -c data/wic/NIKL_SKT_WiC_Train.tsv -t data/wic/NIKL_SKT_WiC_Dev.tsv --model=${model}\
            -o output/gpt2.model --batch_size ${bsz}  --epochs 50 --lr 1e-5 --seed 82
            --input_seq_len 400 --log_freq 1  --accumulate 2"

if [ "${ddp}" = "ddp" ]
then
    cmd="${cmd} --ddp True"
fi

echo $cmd
$cmd


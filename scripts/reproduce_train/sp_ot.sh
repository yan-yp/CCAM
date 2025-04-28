#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# to reproduced the results in our paper, please use:
TRAIN_IMG_SIZE=832
data_cfg_path="configs/data/superglue_trainval_${TRAIN_IMG_SIZE}_5.py"
main_cfg_path="configs/loftr/outdoor/sp_ot.py"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=2
pin_memory=true
exp_name="kp_512"

python -u ./train_sp.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1000 \
    --flush_logs_every_n_steps=1000 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=1 \
    --benchmark=True \
    --max_epochs=30 \
    --resume_from_checkpoint="tf_logs/kp_512/version_3/checkpoints/epoch=27-auc@5=0.281-auc@10=0.456-auc@20=0.617.ckpt"

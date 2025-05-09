#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

train_mode=0
data_cfg_path="configs/data/scannet_test_1500.py"
main_cfg_path="configs/loftr/indoor/scannet/loftr_ds_eval.py"
#main_cfg_path="configs/loftr/indoor/loftr_ds.py"
ckpt_path="logs/MultiScaleLocalFeature_frozen_720/version_3/checkpoints/epoch=25-auc@5=0.456-auc@10=0.621-auc@20=0.753.ckpt"
dump_dir="dump/loftr_ds_indoor"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --train_mode=0
    
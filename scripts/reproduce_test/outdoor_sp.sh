#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/loftr/outdoor/sp_ot_test.py"
#ckpt_path='/media/leo/2t/tf_logs_new/sp_4_2_1_9_ln_noattn_dense2d_k33_mt_layerscale_unet/version_0/checkpoints/epoch=27-auc@5=0.402-auc@10=0.584-auc@20=0.733.ckpt'
#ckpt_path="tf_logs/superglue/version_2/checkpoints/epoch=12-auc@5=0.294-auc@10=0.466-auc@20=0.625.ckpt"
ckpt_path="/home/dulab/zhouyuhang/LoFTR_superglue/superglue_models/weights/superglue_outdoor.pth"
dump_dir="dump/loftr_ot_outdoor"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

python -u ./test_sp.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark

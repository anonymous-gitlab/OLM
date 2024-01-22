
set -x
GPU=$1
model_type=$2
LOG_DIR='./output/'
curr_time=$(date "+%Y_%m_%d_%H_%M_%S")
python -u main_trans_logits.py \
       --dataset='MER2023' \
       --model_type=${model_type} \
       --audio_feature='chinese-hubert-large-FRA' \
       --text_feature='chinese-macbert-large-4-FRA' \
       --video_feature='manet_FRA' \
       --batch_size=2 \
       --lr=1e-4 \
       --gpu=${GPU} \
2>&1 | tee -a ${LOG_DIR}/${curr_time}.log
# model_type: emt, mbt, umt, ca
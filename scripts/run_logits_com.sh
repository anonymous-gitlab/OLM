set -x
GPU=$1
dataset=$2
LOG_DIR='./output/'
curr_time=$(date "+%Y_%m_%d_%H_%M_%S")
python -u main_logits_com.py \
       --dataset=${dataset} \
       --model_type='avc' \
       --fusion_method='bilinear' \
       --batch_size=64 \
       --epochs=10 \
       --lr=1e-4 \
       --gpu=${GPU} \
2>&1 | tee -a ${LOG_DIR}/${curr_time}_${dataset}.log
# model_type: avc
# dataset (optional): AVE, CREMAD, VGGSound, KineticSound
# default batch size: 64
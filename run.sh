#!/usr/bin/env bash

############################################################################################################################
# Script to demonstrate sequence to run for IVDN training
#
# Notes:
# - Full hyperparameter optimisation is skipped right now, such that models trained with optimal params only
# - To run with hyperparameter optimisation, see the "EDIT ME!" sections at the top of each script it the seqeuence
############################################################################################################################
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8

gpu='0';
dataset='brightkite';
seed=0;
pre_batch_size=1000;#1200
model_name=sasrec;
state_iv=True;
state_rmsn=True;
load_model=True;
pre_epochs=100;
epoch_num=300;
treatment_neg_num=100;
train_bsz=2000;
eval_bsz=4000;
test_bsz=4000;
features=64;
output_dir='brightkite/';   #"$dataset"/

config_file="./"$dataset".json"

time_stamp=`date '+%s'`

#commit_id=`git rev-parse HEAD`

echo Start training

std_file=${output_dir}"stdout/"${time_stamp}".txt"
mkdir -p $output_dir"stdout/"

nohup python -u ./main_pretrain.py --gpu=$gpu --data_name ${dataset} --output_dir ${output_dir} --pre_batch_size ${pre_batch_size} --treatment_neg_num ${treatment_neg_num} --pre_epochs ${pre_epochs} --state_iv ${state_iv} --load_model ${load_model} --model_name=$model_name --features=$features --hidden_size=$features --config=$config_file --ts=$time_stamp --dir=$output_dir"stdout/" >>$std_file 2>&1 &

nohup python -u ./main_finetune.py --gpu=$gpu --epoch_num ${epoch_num} --load_model True --state_iv True --model_name sasrec --data_name ${dataset} --output_dir st3 --train_bsz ${train_bsz} --eval_bsz ${eval_bsz} --test_bsz ${test_bsz} --config=$config_file --ts=$time_stamp --dir=$output_dir"stdout/" >>$std_file 2>&1 &

pid=$!

echo "Stdout dir:   $std_file"
echo "Start time:   `date -d @$time_stamp  '+%Y-%m-%d %H:%M:%S'`"
echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
echo "pid:          $pid"
cat $config_file

tail -f $std_file




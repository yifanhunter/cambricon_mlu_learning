#!/bin/bash
set -x
cur_path=$(pwd)
work_dir="${cur_path}/../"
pushd $work_dir
source env.sh

 python run_squad.py \
 --finetune_steps=1000 \
 --hvd_device=mlu \
 --max_seq_length=384 \
 --num_train_epochs=2.0 \
 --doc_stride=128 \
 --save_checkpoints_steps=1000 \
 --do_train=True \
 --do_predict=True \
 --learning_rate=1e-5 \
 --vocab_file="${MODEL_DIR}/vocab.txt" \
 --bert_config_file="${MODEL_DIR}/bert_config.json" \
 --init_checkpoint="${MODEL_DIR}/bert_model.ckpt" \
 --train_file="${SQUAD_DIR}/train-v1.1.json" \
 --predict_file="${SQUAD_DIR}/dev-v1.1.json" \
 --eval_script="${SQUAD_DIR}/evaluate-v1.1.py" \
 --train_batch_size=12 \
 --output_dir=mlu_model \
 --use_profiler=False \
 --use_horovod=False \
 --use_amp=False \
 --use_performance=False

popd

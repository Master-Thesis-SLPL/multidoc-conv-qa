# python ../cqa/run_qa.py \
#  --dataset_name '../utils/quac.py' \
#  --dataset_config_name quac_rc_dev \
#  --model_name_or_path [PATH TO YOUR MODEL] \
#  --do_eval \
#  --logging_steps 2000 \
#  --save_steps 2000 \
#  --num_train_epochs 0 \
#  --max_seq_length 512  \
#  --max_answer_length 100 \
#  --doc_stride 128  \
#  --cache_dir ../../cache \
#  --output_dir [PATH TO YOUR MODEL]/testdev \
#  --overwrite_output_dir  \
#  --per_device_eval_batch_size 2  \
#  --gradient_accumulation_steps 15  \
#  --fp16 \

python ../cqa/run_qa.py \
 --dataset_name  '../utils/dialdoc.py' \
 --dataset_config_name doc2dial_rc_testdev \
 --model_name_or_path '/content/drive/MyDrive/experiments/01-dialdoc/save/checkpoint-2000' \
 --do_eval \
 --logging_steps 2000 \
 --save_steps 2000 \
 --num_train_epochs 0 \
 --max_seq_length 512  \
 --max_answer_length 100 \
 --doc_stride 128  \
 --cache_dir /content/drive/MyDrive/experiments/01-dialdoc/cache \
 --output_dir /content/drive/MyDrive/experiments/01-dialdoc/save \
 --overwrite_output_dir  \
 --per_device_eval_batch_size 2  \
 --gradient_accumulation_steps 15  \
 --fp16

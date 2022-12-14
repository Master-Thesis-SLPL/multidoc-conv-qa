python run_qa.py  --dataset_name  '../utils/coqa.py' --dataset_config_name coqa_rc  --model_name_or_path roberta-large  --do_train  --do_eval  --early_stop  --early_stopping_patience 3  --version_2_with_negative  --logging_steps 10  --save_steps 10  --learning_rate 3e-5   --num_train_epochs 10  --max_seq_length 512   --max_answer_length 50  --doc_stride 128   --cache_dir cache/extra --output_dir save/roberta-large-coqa-quac-doqa  --overwrite_output_dir   --per_device_train_batch_size 2  --per_device_eval_batch_size 2  --gradient_accumulation_steps 2   --evaluation_strategy steps  --eval_steps 10  --load_best_model_at_end  --early_stopping_patience 3  --metric_for_best_model exact  --warmup_steps 20  --weight_decay 0.01  --fp16
#  --sharded_ddp \

# python run_qa.py \
#  --dataset_name  '../utils/dialdoc.py'\
#  --dataset_config_name doc2dial_rc \
#  --model_name_or_path roberta-large \
#  --do_train \
#  --do_eval \
#  --early_stop \
#  --early_stopping_patience 3 \
#  --version_2_with_negative \
#  --logging_steps 100 \
#  --save_steps 100 \
#  --learning_rate 3e-5  \
#  --num_train_epochs 10 \
#  --max_seq_length 512  \
#  --max_answer_length 50 \
#  --doc_stride 128  \
#  --cache_dir cache/extra\
#  --output_dir save/roberta-large-coqa-quac-doqa \
#  --overwrite_output_dir  \
#  --per_device_train_batch_size 2 \
#  --per_device_eval_batch_size 2 \
#  --gradient_accumulation_steps 30  \
#  --evaluation_strategy steps \
#  --eval_steps 100 \
#  --load_best_model_at_end \
#  --early_stopping_patience 3 \
#  --metric_for_best_model exact \
#  --warmup_steps 1000 \
#  --weight_decay 0.01 \
#  --extra_dataset_name '../utils/coqa.py|../utils/quac.py|../utils/doqa.py' \
#  --extra_dataset_config_name "coqa_rc|quac_rc|doqa_rc" \
#  --fp16 \
# #  --sharded_ddp \



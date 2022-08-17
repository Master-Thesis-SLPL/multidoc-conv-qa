# This runs CQA datasets
#  --dataset_name  '../utils/dialdoc.py'\
#  --dataset_config_name doc2dial_rc \
#  '../utils/coqa.py|../utils/quac.py|../utils/doqa.py' \
#   "coqa_rc|quac_rc|doqa_rc"

# hyperparameters={'epochs': 1,                                    # number of training epochs
# 7                  'train_batch_size': 32,                         # batch size for training
# 8                  'eval_batch_size': 64,                          # batch size for evaluation
# 9                  'learning_rate': 3e-5,                          # learning rate used during training
# 10                  'model_id':'distilbert-base-uncased',           # pre-trained model
# 11                  'fp16': True,                                   # Whether to use 16-bit (mixed) precision training
# 12                  'push_to_hub': True,                            # Defines if we want to push the model to the hub
# 13                  'hub_model_id': 'sagemaker-distilbert-emotion', # The model id of the model to push to the hub
# 14                  'hub_strategy': 'every_save',                   # The strategy to use when pushing the model to the hub
# 15                  'hub_token': HfFolder.get_token()               # HuggingFace token to have permission to push
# 16                 }
# 17 
#  --model_name_or_path '/content/drive/MyDrive/experiments/01-dialdoc/save/checkpoint-2000' \


python ../run_qa.py \
 --dataset_name  '../../utils/dialdoc.py' \
 --dataset_config_name doc2dial_rc \
 --model_name_or_path roberta-large \
 --do_train \
 --do_eval \
 --early_stop \
 --early_stopping_patience 3 \
 --version_2_with_negative \
 --logging_steps 200 \
 --save_steps 1000 \
 --learning_rate 3e-5 \
 --num_train_epochs 3 \
 --max_seq_length 512  \
 --max_answer_length 50 \
 --doc_stride 128  \
 --cache_dir /content/drive/MyDrive/experiments/01-dialdoc/cache \
 --output_dir /content/drive/MyDrive/experiments/01-dialdoc/save \
 --overwrite_output_dir  \
 --per_device_train_batch_size 2 \
 --per_device_eval_batch_size 2 \
 --gradient_accumulation_steps 30  \
 --evaluation_strategy steps \
 --eval_steps 1000 \
 --load_best_model_at_end \
 --early_stopping_patience 3 \
 --metric_for_best_model exact \
 --save_total_limit 3 \
 --warmup_steps 1000 \
 --weight_decay 0.01 \
 --fp16 \
 --push_to_hub true \
 --hub_model_id 'alistvt/01-roberta-dialdoc' \
 --hub_strategy 'every_save' \
 --hub_token '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'
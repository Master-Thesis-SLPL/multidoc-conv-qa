cd /data1/alistvt/tes/
source env/bin/activate
cd multidoc-conv-qa/src/span/runners

for RATIO in $(seq 0 9)
do
    python ../run_qa.py --dataset_name ../../utils/multidialdoc.py \
        --dataset_config_name multidoc2dial_rc \
        --model_name_or_path alistvt/docalog \
        --do_eval true \
        --logging_steps 2000 \
        --save_steps 2000 \
        --num_train_epochs 0 \
        --max_seq_length 512 \
        --max_answer_length 100 \
        --doc_stride 128 \
        --cache_dir cache \
        --output_dir save \
        --overwrite_output_dir true \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 15 \
        --fp16 true \
        --version_2_with_negative true \
        --null_score_diff_threshold 0.$RATIO

    cp save/eval_results.txt ../../../results/unknown/0$RATIO.txt
done

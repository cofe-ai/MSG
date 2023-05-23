schedule_str="--grow_time 5
--hidden_size_start 30
--layer_start 20
--layer_start_2 50
--head_start 40
--intermediate_start 10
"

target_str="--hidden_size_target 1024
--layer_target 12
--layer_target_2 24
--head_target 16
--intermediate_target 4096
"

python -m torch.distributed.run \
--nproc_per_node 8 --nnodes 1 \
run_grow_bert.py \
--dataset_name /home/yaoyiqun/growing_bert/bert_data/static_10000 \
--model_name_or_path bert-large-uncased \
--start_config_path ./configs/start_config_try_large_plan_4.json \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 5e-5 \
--max_train_steps 100 \
--num_warmup_steps 30 \
--output_dir ./outputs_no_trainer/grow_full_large_plan_4_test_open \
--max_seq_length 128 \
--checkpointing_steps 10000 \
--logging_steps 10 \
--with_tracking \
${schedule_str} \
${target_str} \
--report_to tensorboard 

# > logs/grow_full_large_plan_4_test_open.txt 2>&1

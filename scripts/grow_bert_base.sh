schedule_str="--grow_time 5
--hidden_size_start 30
--layer_start 20
--layer_start_2 50
--head_start 40
--intermediate_start 10
"

target_str="--hidden_size_target 768
--layer_target 6
--layer_target_2 12
--head_target 12
--intermediate_target 3072
"

python -m torch.distributed.run \
--nproc_per_node 8 --nnodes 1 \
run_grow_bert.py \
--dataset_name /home/yaoyiqun/growing_bert/bert_data/static_10000 \
--model_name_or_path bert-base-uncased \
--start_config_path ./configs/start_config_base_plan_5.json \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-4 \
--max_train_steps 100 \
--num_warmup_steps 30 \
--output_dir ./outputs_no_trainer/grow_full_base_test_open \
--max_seq_length 128 \
--checkpointing_steps 10000 \
--logging_steps 10 \
--with_tracking \
${schedule_str} \
${target_str} \
--report_to tensorboard 

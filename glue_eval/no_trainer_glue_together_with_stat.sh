python -m torch.distributed.run \
--nproc_per_node 4 --nnodes 1 \
run_glue_together_with_stat.py \
  --model_name_or_path bert-base-uncased \
  --pretrained_local_path /your/save/model/path \
  --max_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 

TASK_NAME=squad
python -m torch.distributed.run \
--nproc_per_node 4 --nnodes 1 \
run_squad_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --pretrained_local_path /your/save/model/path \
  --dataset_name ${TASK_NAME} \
  --max_seq_length 384 \
  --per_device_train_batch_size 3 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --output_dir ./tmp/tmp2/${TASK_NAME}/
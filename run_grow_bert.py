#!/usr/bin/env python
# coding=utf-8

'''Main entry for MSG training.'''

import argparse
import json
import logging
import math
import os
import gc
import copy
from pathlib import Path

import datasets
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoTokenizer,
    BertForPreTraining, 
    BertConfig,
    SchedulerType,
)
from transformers.utils import get_full_repo_name
from preprocess_bert_data import DataCollatorNew
from model_ex.grow_ops_v2 import grow_ops
from model_ex.bert_ex_v2 import BertForPreTrainingEx
from model_ex.utils_ex_v2 import get_scheduler_ex, compute_total_norm
from accelerate.utils.other import extract_model_from_parallel


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--start_config_path",
        type=str,
        help="Local path to start config",
        default="./configs/start_config.json",
        required=False,
    )
    parser.add_argument(
        "--grow_init_strategy",
        type=str,
        help="random or random-interpolate",
        default="random",
        required=False,
    )
    parser.add_argument(
        "--new_block_init_strategy",
        type=str,
        help="random/copy",
        default="copy",
        required=False,
    )
    parser.add_argument(
        "--continue_path",
        type=str,
        help="Local path to continue from",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument( 
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient Clipping")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="interval between evaluations",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    # added 
    parser.add_argument(
        "--logging_steps",
        type=str,
        default=None,
        help="Whether to log every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    # MSG main arguments
    parser.add_argument("--grow_time", type=int, default=5000, help="in how many steps to increase mask to 1")
    parser.add_argument("--hidden_size_start", type=int, default=-100, help="in which step to start growing hidden_size")
    parser.add_argument("--layer_start", type=int, default=-100, help="the first time to grow layer")
    parser.add_argument("--layer_start_2", type=int, default=-100, help="the second time to grow layer")
    parser.add_argument("--layer_start_3", type=int, default=-100, help="the third time to grow layer")
    parser.add_argument("--head_start", type=int, default=-100, help="in which step to start growing head_number")
    parser.add_argument("--intermediate_start", type=int, default=-100, help="in which step to start growing ffn_dim")
    
    parser.add_argument("--hidden_size_target", type=int, default=1024, help="default to be Bert-large")
    parser.add_argument("--layer_target", type=int, default=12, help="the first time to grow layer")
    parser.add_argument("--layer_target_2", type=int, default=24, help="the second time to grow layer")
    parser.add_argument("--layer_target_3", type=int, default=-1, help="the third time to grow layer")
    parser.add_argument("--head_target", type=int, default=16, help="in which step to start growing head_number")
    parser.add_argument("--intermediate_target", type=int, default=4096, help="in which step to start growing ffn_dim")

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def prepare_eval_per_device(mlm_preds, nsp_preds, mlm_labels, nsp_labels):
    mlm_preds, nsp_preds = torch.argmax(mlm_preds,-1), torch.argmax(nsp_preds,-1)
    mlm_labels, nsp_labels = mlm_labels.view(-1), nsp_labels.view(-1)
    mlm_preds, nsp_preds = mlm_preds.view(-1), nsp_preds.view(-1)
    mask = mlm_labels != -100
    return mlm_preds, nsp_preds, mlm_labels, nsp_labels, mask


def new_model_and_optimizer(config, args, len_tokenizer):
    model = BertForPreTrainingEx(config)
    model.resize_token_embeddings(len_tokenizer)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return model, optimizer

def free_mem_reloaded(accelerator):
    accelerator._optimizers = []
    accelerator._models = []
    accelerator.deepspeed_engine_wrapped = None
    gc.collect()
    torch.cuda.empty_cache()


def main():

    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    # normal run
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, step_scheduler_with_optimizer=False,
                              **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        data = load_from_disk("{}".format(args.dataset_name))
    else:
        raise ValueError(
            "Should load a static masked dataset."
        )

    config = BertConfig.from_pretrained(args.start_config_path)
    config_up = copy.deepcopy(config)
    logger.info("Growing from the following start config:")
    logger.info(config, main_process_only=True)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, cache_dir="/home/yaoyiqun/huggingface/models")
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir="/home/yaoyiqun/huggingface/models")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.start_config_path:
        logger.info("Training new model from scratch. Config from {}".format(args.start_config_path))
        if args.continue_path:
            model = BertForPreTraining.from_pretrained(args.continue_path, config=config)
        else:
            model = BertForPreTrainingEx(config)


    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing of the datasets is done outside.
    train_dataset = data["train"]
    eval_dataset = data["test"]

    # Data collator
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    data_collator = DataCollatorNew(tokenizer=tokenizer, padding="max_length", max_length=max_seq_length)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, 
                                 batch_size=args.per_device_eval_batch_size)
    
    # Growth settings
    grow_time = args.grow_time
    dimension_dict = {args.hidden_size_start: args.hidden_size_target,
                      args.head_start: args.head_target,
                      args.intermediate_start: args.intermediate_target,
                      args.layer_start: args.layer_target,
                      args.layer_start_2: args.layer_target_2,
                      args.layer_start_3: args.layer_target_3}

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    all_assigned_grows = [x for x in dimension_dict.keys() if x is not None and x > 0]
    if len(all_assigned_grows) == 0:
        rewind_bool, rewind_step_1, rewind_step_2 = False, None, None
    else:
        rewind_bool = True
        rewind_step_1 = sorted(all_assigned_grows)[0]
        rewind_step_2 = sorted(all_assigned_grows)[1] if len(all_assigned_grows) >= 2 else None

    lr_scheduler = get_scheduler_ex(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        rewind_bool=rewind_bool,
        rewind_step_1=rewind_step_1,
        rewind_step_2=rewind_step_2
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    grow_agent = grow_ops(model)
    grow_agent.mask_to_gpu(model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    logging_steps = args.logging_steps
    if logging_steps is not None and logging_steps.isdigit():
        logging_steps = int(logging_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # evaluation agent for accuracy
    metric=evaluate.load("accuracy")

    # grow op initialization
    grow_agent = grow_ops(model)

    grow_step_count = 0
    
    if accelerator.is_main_process:
        json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=1)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                if completed_steps == args.hidden_size_start:
                    if accelerator.is_main_process:
                        grow_agent.print_all_masks(model)
                        grow_agent.print_all_flags(model)

                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"before grow hidden: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    config_up.hidden_size = dimension_dict[completed_steps]
                    new_model, new_optimizer = new_model_and_optimizer(config_up, args, len(tokenizer))
                    grow_agent.set_grow(model, new_model, "hidden_size", 
                                        dimension_dict[completed_steps], grow_time, 
                                        optimizer, new_optimizer, args)
                    
                    del model, optimizer, outputs, loss
                    free_mem_reloaded(accelerator)
                    model, optimizer = accelerator.prepare(new_model, new_optimizer)
                    grow_agent.mask_to_gpu(model)
                    model.train()
                    lr_scheduler.scheduler.optimizer = optimizer
                    grow_step_count = 0
                    
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"after grow: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    grow_agent.print_all_masks(model)
                    grow_agent.print_all_flags(model)

                elif completed_steps in [args.layer_start, args.layer_start_2, args.layer_start_3]:
                    if accelerator.is_main_process:
                        grow_agent.print_all_masks(model)
                        grow_agent.print_all_flags(model)
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"before grow layer: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")

                    config_up.num_hidden_layers = len(extract_model_from_parallel(model).bert.encoder.layer)
                    new_model, new_optimizer = new_model_and_optimizer(config_up, args, len(tokenizer))
                    grow_agent.set_grow(model, new_model, "layers", dimension_dict[completed_steps], 
                                        grow_time, optimizer, new_optimizer, args)
                    del model, optimizer, outputs, loss
                    free_mem_reloaded(accelerator)
                    model, optimizer = accelerator.prepare(new_model, new_optimizer)
                    config_up.num_hidden_layers = dimension_dict[completed_steps]

                    grow_agent.mask_to_gpu(model)
                    model.train()
                    lr_scheduler.scheduler.optimizer = optimizer
                    grow_step_count = 0
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"after grow: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    grow_agent.print_all_masks(model)
                    grow_agent.print_all_flags(model)

                elif completed_steps == args.head_start:
                    if accelerator.is_main_process:
                        grow_agent.print_all_masks(model)
                        grow_agent.print_all_flags(model)
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"before grow head: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    config_up.num_attention_heads = dimension_dict[completed_steps]
                    new_model, new_optimizer = new_model_and_optimizer(config_up, args, len(tokenizer))
                    grow_agent.set_grow(model, new_model, "heads", dimension_dict[completed_steps], 
                                        grow_time, optimizer, new_optimizer, args)
                    del model, optimizer, outputs, loss
                    free_mem_reloaded(accelerator)
                    model, optimizer = accelerator.prepare(new_model, new_optimizer)
                    grow_agent.mask_to_gpu(model)
                    model.train()
                    lr_scheduler.scheduler.optimizer = optimizer
                    grow_step_count = 0
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"after grow: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    grow_agent.print_all_masks(model)
                    grow_agent.print_all_flags(model)

                elif completed_steps == args.intermediate_start:
                    if accelerator.is_main_process:
                        grow_agent.print_all_masks(model)
                        grow_agent.print_all_flags(model)

                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"before grow intermediate: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    config_up.intermediate_size = dimension_dict[completed_steps]
                    new_model, new_optimizer = new_model_and_optimizer(config_up, args, len(tokenizer))
                    grow_agent.set_grow(model, new_model, "intermediate_size", dimension_dict[completed_steps],
                                         grow_time, optimizer, new_optimizer, args)

                    del model, optimizer, outputs, loss
                    free_mem_reloaded(accelerator)
                    model, optimizer = accelerator.prepare(new_model, new_optimizer)
                    grow_agent.mask_to_gpu(model)
                    model.train()
                    lr_scheduler.scheduler.optimizer = optimizer
                    grow_step_count = 0
                    outputs = model(**batch)
                    loss = outputs.loss
                    logger.info(f"after grow: {loss.cpu().data.item()}")
                    logger.info(f"model size: {grow_agent.count_parameters(model)}")
                    grow_agent.print_all_masks(model)
                    grow_agent.print_all_flags(model)

                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            if not grow_agent.available_to_grow:
                if grow_step_count < grow_time:
                    grow_agent.increase_mask(model, None)
                    grow_step_count += 1
                else:
                    grow_agent.end_grow(model, None)
                    grow_step_count = 0

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (40 < completed_steps < 100 or 6000 < completed_steps < 7000 or 8000 < completed_steps < 9000) and accelerator.is_main_process:
                pass
                # grow_agent.print_all_masks(model)
                # print(loss.detach().float().item())

            if completed_steps == 50:
                # grow_agent.print_all_masks(model)
                # print(model.module.bert.encoder.layer[3].attention.self.key.weight)
                # print([group['lr'] for group in optimizer.param_groups])
                pass
                # print(lr_scheduler.scheduler._last_lr)


            # log by steps
            if args.with_tracking and isinstance(logging_steps, int):
                if completed_steps % logging_steps == 0:
                    accelerator.log(
                        {
                            "train_loss_batch": loss.detach().float().item(),
                            "lr": lr_scheduler.scheduler._last_lr[-1],
                            "epoch_process": completed_steps / args.max_train_steps * total_batch_size,
                        },
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # continue

        # do eval
        if args.eval_every is not None and completed_steps % args.eval_every == 0:
            model.eval()
            losses = []
            mlm_preds_all, nsp_preds_all, mlm_labels_all, nsp_labels_all = [], [], [], []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                    mlm_preds, nsp_preds, mlm_labels, nsp_labels, mask = \
                        prepare_eval_per_device(outputs.prediction_logits, 
                        outputs.seq_relationship_logits, 
                        batch["labels"],
                        batch["next_sentence_label"])

                loss = outputs.loss
                loss, mlm_preds, nsp_preds, mlm_labels, nsp_labels, mask = \
                    accelerator.gather_for_metrics((loss.repeat(args.per_device_eval_batch_size),
                    mlm_preds, nsp_preds, mlm_labels, nsp_labels, mask))

                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                mlm_preds_all.append(mlm_preds[mask])
                nsp_preds_all.append(nsp_preds)
                mlm_labels_all.append(mlm_labels[mask])
                nsp_labels_all.append(nsp_labels)

            losses = torch.cat(losses)
            mlm_preds_all = torch.cat(mlm_preds_all)
            nsp_preds_all = torch.cat(nsp_preds_all)
            mlm_labels_all = torch.cat(mlm_labels_all)
            nsp_labels_all = torch.cat(nsp_labels_all)

            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            # compute accuracy metrics
            mlm_acc = metric.compute(predictions=mlm_preds_all, references=mlm_labels_all)["accuracy"]
            nsp_acc = metric.compute(predictions=nsp_preds_all, references=nsp_labels_all)["accuracy"]

            logger.info(f"epoch {epoch}: perplexity: {perplexity} mlm_acc: {mlm_acc} nsp_acc: {nsp_acc}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss.item(),
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "mlm_acc": mlm_acc,
                        "nsp_acc": nsp_acc,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

        # do the following at the end of every epoch
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    try:
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    except Exception as e:
        exit()


if __name__ == "__main__":
    main()

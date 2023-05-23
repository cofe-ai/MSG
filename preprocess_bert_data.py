# coding=utf-8
'''Preprocess Bert pre-training data, create static masks.'''

import numpy as np
import torch
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import DataCollatorWithPadding
import evaluate
from transformers import set_seed
from typing import List, Dict, Any

class DataCollatorNew(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = {key: [example[key] for example in features] for key in features[0].keys() if key != "labels"}
        batch = self.tokenizer.pad(
            encoded_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        mlm_labels = []
        for example in features:
            exam_len = len(example["labels"])
            mlm_labels.append(example["labels"].copy() + [-100] * (self.max_length - exam_len))

        batch["labels"] = torch.tensor(mlm_labels)
        return batch


def static_mask_single(example, tokenizer):
    labels = np.copy(example)
    probability_matrix = np.full(np.array(example).shape, 0.15)
    special_tokens_mask = tokenizer.get_special_tokens_mask(example, already_has_special_tokens=True)
    special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool)
    probability_matrix[special_tokens_mask] = 0

    masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(np.bool)
    # print(labels, masked_indices)
    labels[~masked_indices] = -100 

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool) & masked_indices
    example_out = np.array(example)
    example_out[indices_replaced] = tokenizer.mask_token_id

    indices_random = (
        np.random.binomial(1, 0.5, size=labels.shape).astype(np.bool) & masked_indices & ~indices_replaced
    )
    random_words = np.random.randint(
        low=0, high=len(tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
    )
    example_out[indices_random] = random_words
    return example_out, labels

def static_mask(examples, tokenizer):
    datasets.config.IN_MEMORY_MAX_SIZE=40000000000
    masked_examples = []
    labels = []
    for example in examples["input_ids"]:
        example_out, label = static_mask_single(example, tokenizer)
        masked_examples.append(example_out)
        labels.append(label)
    return {"input_ids": masked_examples, "labels": labels}


def prepare_data(tokenizer):
    nsp_raw = load_dataset("and111/bert_pretrain_phase1", split="train[:10000]")
    nsp_raw = nsp_raw.train_test_split(test_size=0.01)
    nsp_raw_masked = nsp_raw.map(lambda p: static_mask(p, tokenizer), batched=True)
    print(nsp_raw_masked)
    print(nsp_raw_masked["train"][0])
    print("saving static masked dataset...")
    nsp_raw_masked.save_to_disk("bert_data/static_10000")
    print("loading...")
    reloaded = load_from_disk("bert_data/static_10000")
    print(reloaded)
    print(reloaded["train"][0])

    
def write_1000():
    print("start loading")
    full_data = load_from_disk("bert_data/static")
    print(full_data)
    data_1000 = full_data.filter(lambda example, idx: idx < 1000, with_indices=True)
    print(data_1000)
    data_1000.save_to_disk("bert_data/static_1000")


def shuffle_select():
    full_data = load_from_disk("bert_data/static")
    full_data = full_data.shuffle(seed=42)
    # part_data_train = full_data["train"].select(range(full_data["train"].num_rows // 10))
    # part_data_train = full_data["train"].select(range(100000))
    # part_data_test = full_data["test"].select(range(full_data["test"].num_rows // 10))
    part_data_test = full_data["test"].select(range(1000))
    part_data_test.save_to_disk("bert_data/static_1000_test")


if __name__ == "__main__":
    prepare_data()
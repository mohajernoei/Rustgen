#!/usr/bin/env python3

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", nargs="?", default=None)
    ap.add_argument("--size", dest="size_flag", default=None)
    ap.add_argument("--train_json", default=None)
    ap.add_argument("--eval_json", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--project_root", default=None)

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--prompt_max_length", type=int, default=1024)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)

    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=5e-5)

    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_total_limit", type=int, default=2)

    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=1024)
    ap.add_argument("--min_target_tokens", type=int, default=24)

    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    return ap.parse_args()


args = parse_args()
MODEL_SIZE = str(args.size_flag or args.size or "1.3")
MODEL_ID = f"deepseek-ai/deepseek-coder-{MODEL_SIZE}b-base"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(args.project_root or os.environ.get("PROJECT_ROOT", SCRIPT_DIR))
BASE_DIR = PROJECT_ROOT
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
TRAIN_JSON = args.train_json or os.path.join(BASE_DIR, "prepared_sft", "train_messages.jsonl")
EVAL_JSON = args.eval_json or os.path.join(BASE_DIR, "prepared_sft", "eval_messages.jsonl")
OUTPUT_DIR = args.output_dir or os.path.join(ARTIFACTS_DIR, f"deepseek_{MODEL_SIZE}b_stackrust_lora")

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

PROMPT_KEY = "prompt"
COMPLETION_KEY = "completion"

cpu_count = os.cpu_count() or 1
num_proc = max(1, min(8, cpu_count // 2 if cpu_count > 1 else 1))
bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
)
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "right"


def load_and_trim_dataset(path: str, max_samples: int = None):
    ds = load_dataset("json", data_files=path, split="train")
    if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


train_ds = load_and_trim_dataset(TRAIN_JSON, args.max_train_samples)
eval_ds = load_and_trim_dataset(EVAL_JSON, args.max_eval_samples)


TARGET_MIN = args.min_target_tokens


def encode_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    prompts = [(x or "").rstrip() for x in batch.get(PROMPT_KEY, [])]
    completions = [(x or "") for x in batch.get(COMPLETION_KEY, [])]

    eos_ids = [TOKENIZER.eos_token_id] if TOKENIZER.eos_token_id is not None else []

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    all_target_tokens = []

    for prompt, completion in zip(prompts, completions):
        prompt_ids = TOKENIZER(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=args.prompt_max_length,
        )["input_ids"]

        remaining_for_completion = max(1, args.max_length - len(prompt_ids) - len(eos_ids))

        completion_ids = TOKENIZER(
            completion,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining_for_completion,
        )["input_ids"]

        if len(completion_ids) < TARGET_MIN:
            continue

        input_ids = prompt_ids + completion_ids + eos_ids
        labels = ([-100] * len(prompt_ids)) + completion_ids + eos_ids
        attention_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        all_target_tokens.append(len(completion_ids) + len(eos_ids))

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
        "target_tokens": all_target_tokens,
    }


train_ds = train_ds.map(
    encode_batch,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
    remove_columns=train_ds.column_names,
    desc="Tokenizing train",
)

eval_ds = eval_ds.map(
    encode_batch,
    batched=True,
    batch_size=1000,
    num_proc=num_proc,
    remove_columns=eval_ds.column_names,
    desc="Tokenizing eval",
)


def has_target_tokens(example: Dict[str, List[int]]) -> bool:
    return any(x != -100 for x in example["labels"]) and example.get("target_tokens", 0) >= TARGET_MIN


train_ds = train_ds.filter(has_target_tokens, num_proc=num_proc, desc="Filtering train")
eval_ds = eval_ds.filter(has_target_tokens, num_proc=num_proc, desc="Filtering eval")


@dataclass
class CompletionOnlyCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if bf16_supported else torch.float16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    logging_steps=args.logging_steps,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=args.save_total_limit,
    report_to=[],
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,
    dataloader_num_workers=min(4, cpu_count),
    bf16=bf16_supported,
    fp16=not bf16_supported,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=CompletionOnlyCollator(TOKENIZER),
)

print(f"Train rows after token filtering: {len(train_ds)}")
print(f"Eval rows after token filtering : {len(eval_ds)}")
model.print_trainable_parameters()
trainer.train()

trainer.save_model(OUTPUT_DIR)
TOKENIZER.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to: {OUTPUT_DIR}")

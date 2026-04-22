#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model ID or local path")
    ap.add_argument("--adapter", required=True, help="PEFT/LoRA adapter directory")
    ap.add_argument("--out", required=True, help="Output directory for merged full model")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=None,
    ).to("cuda")

    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = peft_model.merge_and_unload()

    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)

    print(f"Merged model saved to: {args.out}")


if __name__ == "__main__":
    main()

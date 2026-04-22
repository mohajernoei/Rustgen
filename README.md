# DeepSeek Rust SFT + Merge + MultiPL-E Evaluation

This project automates the **full cycle** for Rust code fine-tuning and evaluation:

1. clone or update MultiPL-E,
2. download the dataset into the project’s local `data/` folder if it is missing,
3. prepare completion-style SFT examples,
4. fine-tune a DeepSeek Coder base model with LoRA,
5. merge the adapter into a full model,
6. evaluate the merged model with MultiPL-E on HumanEval-RS.

The project folder name is **agnostic**. The scripts use the directory containing `run.sh` as the default `PROJECT_ROOT`, so you can rename the project folder freely.

---

## 1) What `run.sh` does now

The main entrypoint is:

```bash
bash run.sh 1.3
```

When you run it, it will automatically:

- use the script folder as `PROJECT_ROOT`,
- create local project folders such as `data/`, `prepared_sft/`, and `artifacts/`,
- clone `MultiPL-E` into `PROJECT_ROOT/MultiPL-E-main` if it is missing,
- update `MultiPL-E` if it already exists,
- check whether `data/stack_rust_clean.jsonl` exists,
- **download the dataset only if it does not exist**,
- reuse the dataset if it is already present,
- prepare the SFT train/eval JSONL files,
- train the LoRA adapter,
- merge the adapter into a full model,
- run MultiPL-E generation and evaluation,
- print pass@k results.

So the dataset behavior is now:

- **missing dataset** -> download it into the project’s local `data/` directory
- **existing dataset** -> do not download again

---

## 2) Recommended project layout

Put these files in one folder:

```text
your-project/
├── run.sh
├── run_test.sh
├── datacollector.sh
├── prepare_deepseek_messages.py
├── sft.py
├── merge.py
├── requirements.txt
└── README.md
```

After running `run.sh`, the project will create this structure automatically:

```text
your-project/
├── data/
│   └── stack_rust_clean.jsonl
├── prepared_sft/
│   ├── train_messages.jsonl
│   └── eval_messages.jsonl
├── artifacts/
│   └── deepseek_1.3b_stackrust_lora/
├── MultiPL-E-main/
│   └── tutorial/
│       ├── deepseek-1.3b-stackrust-merged/
│       ├── humaneval-rs-deepseek_ai_deepseek_coder_1.3b_base-0.8-reworded/
│       └── humaneval-rs-deepseek_1.3b_stackrust_merged-0.8-reworded/
```

---

## 3) Prerequisites

Install system tools first.

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y git curl jq podman python3 python3-pip
```

You also need:

- an NVIDIA GPU for practical training,
- a working CUDA/PyTorch setup,
- enough disk space for model weights, artifacts, and evaluation outputs,
- internet access for Hugging Face downloads, dataset download, container pull, and the MultiPL-E clone.

---

## 4) Python environment

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Make scripts executable

```bash
chmod +x run.sh run_test.sh datacollector.sh
```

---

## 6) One-command full automation

This is the main command:

```bash
bash run.sh 1.3
```

For the 6.7B model:

```bash
bash run.sh 6.7
```

That single command runs the whole cycle:

1. ensure `MultiPL-E-main/` exists,
2. pull/update MultiPL-E,
3. check `data/stack_rust_clean.jsonl`,
4. download the dataset only if it is missing,
5. build `prepared_sft/train_messages.jsonl` and `prepared_sft/eval_messages.jsonl`,
6. train the LoRA adapter,
7. merge the adapter into a full model,
8. run HumanEval-RS generation and evaluation,
9. compute pass@k.

---

## 7) Where the dataset goes

`run.sh` uses the project’s local `data/` directory by default:

```text
data/stack_rust_clean.jsonl
```

That means:

- if the dataset file is **not** there, `run.sh` calls `datacollector.sh`
- if the dataset file **is** already there, `run.sh` skips the download

You do **not** need to run `datacollector.sh` manually unless you want to.

---

## 8) Manual dataset download only

If you want to download the dataset without running training/evaluation yet:

```bash
bash datacollector.sh
```

This writes to:

```text
data/stack_rust_clean.jsonl
```

The data collector is project-root aware, so it saves into the local project folder rather than depending on the shell’s current directory.

---

## 9) Smoke test run

Before a large run, you can do a lighter smoke test:

```bash
MAX_SAMPLES=2000 \
MAX_EVAL_SAMPLES=64 \
COMPLETION_LIMIT=1 \
RUN_BASELINE=0 \
bash run.sh 1.3
```

This still runs the full cycle, but with a much smaller dataset slice and cheaper evaluation.

---

## 10) Important environment variables

You can customize behavior without editing the files.

### Common variables

```bash
export PROJECT_ROOT=/absolute/path/to/your-project
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=your_token_here
export RUN_BASELINE=1
export COMPLETION_LIMIT=10
export TEMPERATURE=0.8
export MAX_SAMPLES=250000
export MAX_EVAL_SAMPLES=2048
```

### What they mean

- `PROJECT_ROOT`: overrides the default project folder
- `CUDA_VISIBLE_DEVICES`: selects GPU(s)
- `HF_TOKEN`: optional Hugging Face token for dataset/model access
- `RUN_BASELINE=1`: also evaluate the base model for comparison
- `COMPLETION_LIMIT`: number of completions per task in MultiPL-E
- `TEMPERATURE`: generation temperature for evaluation
- `MAX_SAMPLES`: cap dataset size used for SFT
- `MAX_EVAL_SAMPLES`: cap number of eval rows used during data prep / training stage

---

## 11) Manual step-by-step commands

If you want to run the stages individually instead of using the one-command automation:

### Step 1: Download dataset

```bash
bash datacollector.sh
```

### Step 2: Prepare SFT examples

```bash
python prepare_deepseek_messages.py \
  --input_jsonl data/stack_rust_clean.jsonl \
  --out_dir prepared_sft \
  --eval_fraction 0.01 \
  --min_chars 120 \
  --min_prompt_chars 48 \
  --min_completion_chars 48 \
  --max_samples 250000 \
  --max_eval_samples 2048 \
  --max_chars 24000 \
  --max_examples_per_file 3
```

### Step 3: Train the adapter

```bash
python sft.py 1.3 \
  --project_root . \
  --train_json prepared_sft/train_messages.jsonl \
  --eval_json prepared_sft/eval_messages.jsonl \
  --output_dir artifacts/deepseek_1.3b_stackrust_lora \
  --max_length 2048 \
  --prompt_max_length 1024 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --logging_steps 20 \
  --save_total_limit 2 \
  --max_eval_samples 2048 \
  --min_target_tokens 24 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05
```

### Step 4: Merge adapter

```bash
python merge.py \
  --base deepseek-ai/deepseek-coder-1.3b-base \
  --adapter artifacts/deepseek_1.3b_stackrust_lora \
  --out MultiPL-E-main/tutorial/deepseek-1.3b-stackrust-merged \
  --dtype bf16
```

### Step 5: Evaluate

```bash
bash run_test.sh 1.3
```

---

## 12) Outputs you should expect

### Dataset

```text
data/stack_rust_clean.jsonl
```

### Prepared SFT files

```text
prepared_sft/train_messages.jsonl
prepared_sft/eval_messages.jsonl
```

### LoRA adapter

```text
artifacts/deepseek_1.3b_stackrust_lora/
```

### Merged model

```text
MultiPL-E-main/tutorial/deepseek-1.3b-stackrust-merged/
```

### MultiPL-E evaluation outputs

Examples:

```text
MultiPL-E-main/tutorial/humaneval-rs-deepseek_ai_deepseek_coder_1.3b_base-0.8-reworded/
MultiPL-E-main/tutorial/humaneval-rs-deepseek_1.3b_stackrust_merged-0.8-reworded/
```

These folders will contain generations and evaluation result files used by `pass_k.py`.

---

## 13) Troubleshooting

### Dataset does not download

Check that these are installed:

```bash
curl
jq
```

Install them on Ubuntu/Debian with:

```bash
sudo apt update
sudo apt install -y curl jq
```

### MultiPL-E does not clone/update

Check that `git` is installed and that your internet connection is available.

### Podman evaluation fails

Make sure `podman` is installed and working:

```bash
podman --version
```

### Training fails on GPU memory

Try a smaller run first:

```bash
MAX_SAMPLES=2000 MAX_EVAL_SAMPLES=64 RUN_BASELINE=0 COMPLETION_LIMIT=1 bash run.sh 1.3
```

You can also reduce memory pressure by adjusting batch size or using a smaller model.

### Hugging Face access issues

Set a token in the shell before running:

```bash
export HF_TOKEN=your_token_here
```

---

## 14) Recommended first command

If your environment is already set up, this is the best first run:

```bash
MAX_SAMPLES=2000 MAX_EVAL_SAMPLES=64 COMPLETION_LIMIT=1 RUN_BASELINE=0 bash run.sh 1.3
```

Then, when that succeeds, run the full version:

```bash
bash run.sh 1.3
```

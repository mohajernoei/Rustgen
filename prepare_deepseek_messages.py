#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

RUST_FN_RE = re.compile(
    r'(?ms)^(?P<indent>[ \t]*)(?:(?:pub(?:\([^\)]*\))?|async|const|unsafe|default|extern\s+"[^"]+"|extern|crate)\s+)*fn\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:<[^\n\{\)]*>)?\s*\([^\)]*\)\s*(?:->\s*[^\{\n]+)?\s*\{'
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Prepare higher-quality Rust completion-style JSONL files from a local JSONL dataset."
    )
    ap.add_argument("--input_jsonl", required=True, help="Input JSONL file with code samples.")
    ap.add_argument("--out_dir", required=True, help="Output directory for train/eval JSONL.")
    ap.add_argument("--eval_fraction", type=float, default=0.01, help="Fraction for eval split.")
    ap.add_argument("--min_chars", type=int, default=120, help="Minimum code length.")
    ap.add_argument("--min_prompt_chars", type=int, default=48, help="Minimum prompt length after splitting.")
    ap.add_argument("--min_completion_chars", type=int, default=48, help="Minimum completion length.")
    ap.add_argument("--max_samples", type=int, default=250000, help="Maximum total number of samples to keep after shuffle.")
    ap.add_argument("--max_eval_samples", type=int, default=2048, help="Maximum eval samples.")
    ap.add_argument("--max_chars", type=int, default=24000, help="Discard very large source files.")
    ap.add_argument("--max_examples_per_file", type=int, default=3, help="Maximum examples to extract per input file.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


SKIP_SUBSTRINGS = (
    "Copyright",
    "Licensed under",
    "SPDX-License-Identifier",
    "generated automatically",
)


RUST_HINTS = (
    "fn ",
    "impl ",
    "struct ",
    "enum ",
    "trait ",
    "use ",
    "let ",
    "match ",
    "pub ",
    "mod ",
)


def extract_code(obj: Dict) -> Optional[str]:
    for key in ("code", "text", "content", "completion"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def looks_like_rust(code: str) -> bool:
    hits = sum(1 for hint in RUST_HINTS if hint in code)
    return hits >= 2 and ("fn " in code or "impl " in code or "struct " in code)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_noise(code: str) -> str:
    code = normalize_newlines(code).strip()
    lines = code.split("\n")
    filtered = []
    for line in lines:
        if any(token in line for token in SKIP_SUBSTRINGS):
            continue
        filtered.append(line.rstrip())
    code = "\n".join(filtered).strip()
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code


def find_matching_brace(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_str = False
    in_char = False
    in_line_comment = False
    in_block_comment = 0
    escape = False

    for i in range(open_idx, len(text)):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue

        if in_block_comment:
            if ch == "/" and nxt == "*":
                in_block_comment += 1
                continue
            if ch == "*" and nxt == "/":
                in_block_comment -= 1
                continue
            continue

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if in_char:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_char = False
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = 1
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "'":
            in_char = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

    return None


def choose_prompt_start(code: str, fn_start: int) -> int:
    window_start = max(0, fn_start - 2200)
    candidate = code.rfind("\n\n", window_start, fn_start)
    if candidate != -1 and fn_start - candidate >= 80:
        return candidate + 2
    return window_start


def make_examples_from_code(
    code: str,
    min_prompt_chars: int,
    min_completion_chars: int,
    max_examples_per_file: int,
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []

    for match in RUST_FN_RE.finditer(code):
        fn_start = match.start()
        open_brace = code.find("{", match.end() - 1)
        if open_brace == -1:
            continue
        close_brace = find_matching_brace(code, open_brace)
        if close_brace is None:
            continue

        prompt_start = choose_prompt_start(code, fn_start)
        prompt = code[prompt_start : open_brace + 1].rstrip()
        completion = code[open_brace + 1 : close_brace + 1].lstrip("\n")

        if len(prompt) < min_prompt_chars or len(completion) < min_completion_chars:
            continue

        if completion.count("\n") < 2:
            continue

        examples.append({
            "prompt": prompt,
            "completion": completion,
        })

        if len(examples) >= max_examples_per_file:
            break

    if not examples:
        split_markers: List[int] = []
        for marker in ("\nimpl ", "\nstruct ", "\nenum ", "\ntrait ", "\nmod "):
            idx = code.find(marker, max(min_prompt_chars, len(code) // 6))
            if idx != -1:
                split_markers.append(idx + 1)
        split_markers.append(max(min_prompt_chars, int(len(code) * 0.45)))
        split_markers.append(max(min_prompt_chars, int(len(code) * 0.6)))

        for split_at in sorted(set(split_markers)):
            if split_at >= len(code) - min_completion_chars:
                continue
            prompt = code[:split_at].rstrip()
            completion = code[split_at:].lstrip()
            if len(prompt) >= min_prompt_chars and len(completion) >= min_completion_chars:
                examples.append({"prompt": prompt, "completion": completion})
                break

    return examples


def dedupe_examples(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for row in rows:
        key = (
            re.sub(r"\s+", " ", row["prompt"]).strip(),
            re.sub(r"\s+", " ", row["completion"]).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_records(
    path: Path,
    min_chars: int,
    min_prompt_chars: int,
    min_completion_chars: int,
    max_chars: int,
    max_examples_per_file: int,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    raw_seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            code = extract_code(obj)
            if not code:
                continue

            code = strip_noise(code)
            if len(code) < min_chars or len(code) > max_chars:
                continue
            if code in raw_seen:
                continue
            if not looks_like_rust(code):
                continue
            if "unsafe impl" in code and code.count("fn ") == 0:
                continue

            rows = make_examples_from_code(
                code,
                min_prompt_chars=min_prompt_chars,
                min_completion_chars=min_completion_chars,
                max_examples_per_file=max_examples_per_file,
            )
            if not rows:
                continue

            raw_seen.add(code)
            records.extend(rows)

    return dedupe_examples(records)


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(
        input_path,
        min_chars=args.min_chars,
        min_prompt_chars=args.min_prompt_chars,
        min_completion_chars=args.min_completion_chars,
        max_chars=args.max_chars,
        max_examples_per_file=args.max_examples_per_file,
    )
    if not records:
        raise ValueError("No valid Rust completion samples were created from the input JSONL.")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    original_count = len(records)
    if args.max_samples and args.max_samples > 0:
        records = records[: min(args.max_samples, len(records))]

    eval_count = max(1, int(len(records) * args.eval_fraction))
    if args.max_eval_samples and args.max_eval_samples > 0:
        eval_count = min(eval_count, args.max_eval_samples)

    eval_rows = records[:eval_count]
    train_rows = records[eval_count:]
    if not train_rows:
        raise ValueError("No training rows left after eval split. Increase max_samples or reduce eval_fraction.")

    train_path = out_dir / "train_messages.jsonl"
    eval_path = out_dir / "eval_messages.jsonl"

    train_n = write_jsonl(train_path, train_rows)
    eval_n = write_jsonl(eval_path, eval_rows)

    print(f"Loaded high-quality samples : {original_count}")
    print(f"Kept total samples          : {len(records)}")
    print(f"Train samples               : {train_n}")
    print(f"Eval samples                : {eval_n}")
    print(f"Wrote                       : {train_path}")
    print(f"Wrote                       : {eval_path}")


if __name__ == "__main__":
    main()

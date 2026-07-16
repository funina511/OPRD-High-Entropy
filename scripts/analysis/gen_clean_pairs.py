#!/usr/bin/env python3
"""Generate coherent TEACHER solutions for bridge fitting, on the SAME prompts as the
dirty on-policy pairs.

Why: the base-student on-policy pairs are ~25% garbled (base @ T=1.0 on hard math), so
PCA on teacher hidden states over those sequences puts high-variance
"language-drift / repetition" directions into P_T. We instead regenerate responses with
the TEACHER (Qwen3-4B, coherent) on the *identical* prompt set, so the ONLY variable vs
the dirty bridge is response text quality — NOT the problem distribution.

We keep ALL teacher solutions (correct or not): a wrong-but-coherent reasoning chain is
still clean signal for PCA (coherence matters, not final correctness). Gold correctness
is recorded as metadata for diagnostics only, never used to filter.

Output: {raw_prompt, response, teacher_correct} JSONL — bridge trainer reads the first two.
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl" / "verl"))
from utils.reward_score.ttrl_math.math_utils import (  # noqa: E402
    extract_boxed_answer, grade_answer_mathd, grade_answer_sympy,
)


def extract_pred(text: str):
    if "\\boxed" in text:
        b = extract_boxed_answer(text)
        if b:
            return b
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    return None


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return bool(grade_answer_mathd(pred, gold) or grade_answer_sympy(pred, gold))
    except Exception:
        return False


def prompt_key(raw_prompt):
    """Join user-message contents to a stable string key for gold lookup."""
    return "\n".join(m["content"] for m in raw_prompt if m.get("role") == "user")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model-path", required=True)
    ap.add_argument("--dirty-pairs", required=True, help="reuse this file's raw_prompts (exact same problems)")
    ap.add_argument("--data-parquet", required=True, help="source of gold answers (for diagnostics only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    # 1) exact same prompts as the dirty bridge
    dirty = [json.loads(l) for l in open(args.dirty_pairs, encoding="utf-8")]
    prompts = [r["raw_prompt"] for r in dirty]
    print(f"Reusing {len(prompts)} prompts from {args.dirty_pairs}", flush=True)

    # 2) gold map from parquet (content -> ground_truth), diagnostics only
    df = pd.read_parquet(args.data_parquet)
    gold_map = {}
    for p, rm in zip(df["prompt"].tolist(), df["reward_model"].tolist()):
        msgs = [{"content": m["content"], "role": m["role"]} for m in list(p)]
        gold_map[prompt_key(msgs)] = str(rm["ground_truth"])

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    llm = LLM(model=args.teacher_model_path, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=6144, dtype="bfloat16")
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens)

    # enable_thinking=False = Qwen3 official HARD switch (injects empty <think></think>)
    texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False) for m in prompts]
    print(f"Generating 1 teacher solution/prompt (T={args.temperature}, thinking OFF)...", flush=True)
    outs = llm.generate(texts, sp)

    kept, n_correct, n_gold = [], 0, 0
    for raw_prompt, out in zip(prompts, outs):
        r = out.outputs[0].text
        gold = gold_map.get(prompt_key(raw_prompt))
        ok = None
        if gold is not None:
            n_gold += 1
            ok = is_correct(extract_pred(r), gold)
            n_correct += int(ok)
        kept.append({"raw_prompt": raw_prompt, "response": r, "teacher_correct": ok})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    acc = f"{n_correct}/{n_gold}" if n_gold else "n/a (gold unmatched)"
    print(f"kept={len(kept)} (ALL) | teacher_acc={acc} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

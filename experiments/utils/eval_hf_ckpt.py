"""Evaluate merged HF checkpoint(s) on math/reasoning datasets.

Loads one or more trained-and-merged HF checkpoints, samples N rollouts per
question with vLLM (one model per GPU, rollouts round-robin across GPUs), and
grades with the rule-based verifier (grade_answer_verl). Reports mean@k / best@k /
noboxed_frac per (arm, task) and writes per-(arm,task) jsonl + summary.json.

Defaults align with OPRD training val: temp=0.7, top_p=0.95, max_tokens=8192,
enable_thinking=False, chat template.

CLI:
  # single merged ckpt on default tasks (MATH-500/AIME24/AIME25):
  python eval_hf_ckpt.py --model /path/to/merged_ckpt --name myrun

  # multiple arms + explicit tasks + custom GPUs:
  python eval_hf_ckpt.py --arms A:/ckptA,B:/ckptB --tasks MATH-500,AMC23 --gpus 0,1,2,3

  # arbitrary parquet not in the registry (name=path:N):
  python eval_hf_ckpt.py --model /ckpt --tasks-spec mydata=/abs/test.parquet:8

Import:
  from eval_hf_ckpt import evaluate, DEFAULT_TASKS
  summary = evaluate({"A": "/ckptA"}, tasks=["MATH-500"], outdir="out")
"""
import os, sys, json, argparse, multiprocessing, gc
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# grade_answer_verl lives in scripts/val/eval.
_EVAL_UTILS_DIR = os.environ.get(
    "OPRD_EVAL_UTILS_DIR", "/mnt/lxy/OPRD-High-Entropy/scripts/val/eval")
if _EVAL_UTILS_DIR not in sys.path:
    sys.path.insert(0, _EVAL_UTILS_DIR)

# ---- defaults (align with training val) ----
TEMPERATURE, TOP_P, MAX_TOKENS = 0.7, 0.95, 8192
ENABLE_THINKING = False
PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
DATA_ROOT = os.environ.get("OPRD_TEST_DATA", "/mnt/lxy/OPRD-High-Entropy/datasets/test_data")

# task name -> default N (rollouts). Any dir under DATA_ROOT with test.parquet works.
DEFAULT_N = {"MATH-500": 4, "AIME24": 16, "AIME25": 16, "AMC23": 8}
DEFAULT_TASKS = ["MATH-500", "AIME24", "AIME25"]


class EvalConfig:
    """Runtime knobs; override any field via kwargs to evaluate()."""
    def __init__(self, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                 enable_thinking=ENABLE_THINKING, prompt_template=PROMPT_TEMPLATE,
                 gpus=None, gpu_mem_util=0.85, trust_remote_code=True,
                 apply_chat_template=True):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.prompt_template = prompt_template
        self.gpus = gpus or [int(x) for x in os.environ.get(
            "CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",") if x != ""]
        self.gpu_mem_util = gpu_mem_util
        self.trust_remote_code = trust_remote_code
        self.apply_chat_template = apply_chat_template


def resolve_task(name, spec=None):
    """Return {'name','path','N'}. `spec` = 'path:N' overrides the registry."""
    if spec:
        path, _, n = spec.partition(":")
        return {"name": name, "path": path, "N": int(n or DEFAULT_N.get(name, 8))}
    return {"name": name, "path": f"{DATA_ROOT}/{name}/test.parquet",
            "N": DEFAULT_N.get(name, 8)}


def load_samples(path):
    df = pd.read_parquet(path)
    return [
        {"example_id": i,
         "prompt": df.at[i, "prompt"][0]["content"].strip(),
         "answer": df.at[i, "reward_model"]["ground_truth"].strip()}
        for i in range(len(df))
    ]


def _worker(t):
    model, samples, roll_ids, gpu, cfg = t
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from vllm import LLM, SamplingParams
    out = []
    llm = LLM(model=model, trust_remote_code=cfg.trust_remote_code,
              gpu_memory_utilization=cfg.gpu_mem_util, tensor_parallel_size=1,
              max_model_len=cfg.max_tokens + 1024)
    tok = llm.get_tokenizer()
    stop_ids = []
    for s in ["<|im_end|>", "<|endoftext|>"]:
        e = tok.encode(s, add_special_tokens=False)
        if e:
            stop_ids.append(e[0])
    sp = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                        max_tokens=cfg.max_tokens, stop_token_ids=stop_ids or None)
    prompts = []
    for s in samples:
        text = cfg.prompt_template.format(problem=s["prompt"])
        if cfg.apply_chat_template:
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=cfg.enable_thinking)
            except TypeError:  # tokenizers without enable_thinking kwarg
                text = tok.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False,
                    add_generation_prompt=True)
        prompts.append(text)
    for rid in roll_ids:
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for s, o in zip(samples, outs):
            out.append({"example_id": s["example_id"], "answer": s["answer"],
                        "seed": rid, "response": o.outputs[0].text})
    del llm; gc.collect()
    return out


def gen_task(model, task, out_path, cfg):
    samples = load_samples(task["path"])
    gpus = cfg.gpus
    chunks = [[] for _ in gpus]
    for i in range(task["N"]):
        chunks[i % len(gpus)].append(i)
    args = [(model, samples, chunks[i], gpus[i], cfg)
            for i in range(len(gpus)) if chunks[i]]
    res = []
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(args), mp_context=ctx) as ex:
        futs = [ex.submit(_worker, a) for a in args]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc=task["name"]):
            res.extend(f.result())
    with open(out_path, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(res)


def grade(jsonl_path):
    from utils import grade_answer_verl
    by_id = {}
    for line in open(jsonl_path):
        d = json.loads(line)
        by_id.setdefault(d["example_id"], {"gt": d["answer"], "resp": []})["resp"].append(d["response"])
    means, bests, nolen = [], [], 0
    for _id, v in by_id.items():
        sc = [bool(grade_answer_verl(r, v["gt"])) for r in v["resp"]]
        means.append(sum(sc) / len(sc))
        bests.append(1.0 if any(sc) else 0.0)
        nolen += sum("boxed" not in r for r in v["resp"])
    n = len(by_id)
    total_resp = sum(len(v["resp"]) for v in by_id.values())
    return {"n_q": n, "mean@k": sum(means)/n, "best@k": sum(bests)/n,
            "maj_solved_all": sum(m == 1 for m in means),
            "solved_none": sum(m == 0 for m in means),
            "noboxed_frac": nolen / total_resp}


def evaluate(arms, tasks=None, task_specs=None, outdir="eval_out", cfg=None, **cfg_kw):
    """arms: {name: model_path}. tasks: list of registry names. task_specs:
    {name: 'path:N'} for arbitrary parquet. Returns summary dict; also writes
    per-(arm,task) jsonl + summary.json under outdir. Skips generation when the
    jsonl already exists (resume-friendly)."""
    cfg = cfg or EvalConfig(**cfg_kw)
    tasks = tasks or DEFAULT_TASKS
    task_specs = task_specs or {}
    resolved = [resolve_task(t, task_specs.get(t)) for t in tasks]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm_name, model in arms.items():
        summary[arm_name] = {}
        for task in resolved:
            jp = outdir / f"{arm_name}__{task['name']}_n{task['N']}.jsonl"
            if not jp.exists():
                print(f"[GEN] {arm_name} / {task['name']}", flush=True)
                gen_task(model, task, jp, cfg)
            summary[arm_name][task["name"]] = grade(jp)
            print(f"[GRADED] {arm_name}/{task['name']}: {summary[arm_name][task['name']]}", flush=True)
        with open(outdir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="single merged HF ckpt path (pair with --name)")
    g.add_argument("--arms", help="comma list name:path,name:path")
    ap.add_argument("--name", default="model", help="arm name when using --model")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma registry names")
    ap.add_argument("--tasks-spec", default="", help="comma name=path:N for arbitrary parquet")
    ap.add_argument("--outdir", default="/mnt/lxy/OPRD-High-Entropy/logs/eval_hf")
    ap.add_argument("--gpus", default="", help="comma GPU ids (default: all visible)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=TOP_P)
    ap.add_argument("--no-chat-template", action="store_true")
    return ap.parse_args()


def main():
    a = _parse_args()
    arms = dict(x.split(":", 1) for x in a.arms.split(",")) if a.arms else {a.name: a.model}
    tasks = [t for t in a.tasks.split(",") if t]
    task_specs = {}
    for kv in [x for x in a.tasks_spec.split(",") if x]:
        name, _, spec = kv.partition("=")
        task_specs[name] = spec
        if name not in tasks:
            tasks.append(name)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""] or None
    evaluate(arms, tasks=tasks, task_specs=task_specs, outdir=a.outdir,
             gpus=gpus, max_tokens=a.max_tokens, temperature=a.temperature,
             top_p=a.top_p, apply_chat_template=not a.no_chat_template)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

import re, json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHAIN = "logs/chain_opd_surf_n8_20260721_190309.log"
VAL = {
    "OPD (RKL)": "outputs/logs/validation_log/cmp_opd_rkl_n8_r8192_20260721_190309",
    "SURF":      "outputs/logs/validation_log/cmp_surf_n8_r8192_20260721_190309",
}

def parse_train(path):
    """Split chain log into runs on step reset; return list of {step: {ent,len}}."""
    runs, cur, last = [], None, 1e9
    for line in open(path, errors="ignore"):
        m = re.search(r"step:(\d+)\b", line)
        if not m:
            continue
        s = int(m.group(1))
        if cur is None or s < last:
            cur = {}; runs.append(cur)
        last = s
        d = cur.setdefault(s, {})
        e = re.search(r"actor/entropy:([\-\d.eE]+)", line)
        l = re.search(r"response_length/mean:([\d.]+)", line)
        if e: d["ent"] = float(e.group(1))
        if l: d["len"] = float(l.group(1))
    return runs


def val_acc(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        step = int(re.search(r"(\d+)\.jsonl", f).group(1))
        n = c = 0
        for line in open(f, errors="ignore"):
            try: r = json.loads(line)
            except: continue
            n += 1; c += 1 if r.get("acc") else 0
        if n: out[step] = c / n
    return dict(sorted(out.items()))


runs = parse_train(CHAIN)
# run 0 = OPD, run 1 = SURF (order in chain log)
train = {"OPD (RKL)": runs[0], "SURF": runs[1]}


COL = {"OPD (RKL)": "#1b7f4d", "SURF": "#c0392b"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# ---- Left: entropy (left y) + val_acc (right y) ----
axL = axes[0]; axLr = axL.twinx()
for name in ["OPD (RKL)", "SURF"]:
    tr = train[name]
    xs = sorted(tr)
    es = [tr[s].get("ent") for s in xs]
    xs2 = [s for s, e in zip(xs, es) if e is not None]
    es2 = [e for e in es if e is not None]
    axL.plot(xs2, es2, color=COL[name], lw=1.8, label=f"{name} entropy")
    va = val_acc(VAL[name])
    axLr.plot(list(va), list(va.values()), color=COL[name], lw=2.4,
              ls="--", marker="o", ms=5, label=f"{name} val_acc")

axL.set_xlabel("training step"); axL.set_ylabel("actor entropy (solid)")
axLr.set_ylabel("val accuracy (dashed)")
axL.set_yscale("log")
axL.set_title("Entropy collapse vs. accuracy collapse")
axL.axhline(0.04, color="gray", ls=":", lw=1)
axL.text(2, 0.043, "entropy ≈ 0.04", color="gray", fontsize=8)
hL = axL.get_legend_handles_labels()[0] + axLr.get_legend_handles_labels()[0]
lL = axL.get_legend_handles_labels()[1] + axLr.get_legend_handles_labels()[1]
axL.legend(hL, lL, fontsize=8, loc="center right")

# ---- Right: response length ----
axR = axes[1]
for name in ["OPD (RKL)", "SURF"]:
    tr = train[name]
    xs = sorted(tr)
    ls = [tr[s].get("len") for s in xs]
    xs2 = [s for s, l in zip(xs, ls) if l is not None]
    ls2 = [l for l in ls if l is not None]
    axR.plot(xs2, ls2, color=COL[name], lw=1.8, label=name)
axR.axhline(8192, color="gray", ls=":", lw=1); axR.text(2, 8300, "max=8192", color="gray", fontsize=8)
axR.set_xlabel("training step"); axR.set_ylabel("response length (tokens)")
axR.set_title("Response length"); axR.legend(fontsize=9)

fig.suptitle("OPD (RKL) vs SURF — same teacher(Qwen3-4B)/student(Qwen3-0.6B-Base)/n=8/r8192",
             fontsize=12, y=1.02)
fig.tight_layout()
out = "outputs/opd_vs_surf_entropy_acc.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)

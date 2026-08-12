#!/usr/bin/env python3
"""STARXION v1 — silent-compute integrity proof of concept.

Software-injected fault campaign on a tiny PyTorch Transformer-style model.
This is not radiation testing or flight-qualified software.
"""
from __future__ import annotations
import json, time, html
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(20260812)
torch.set_num_threads(1)

CLEAN_TRIALS = 300
SINGLE_TRIALS = 1000
DOUBLE_TRIALS = 500
FFN_TRIALS = 500


class GuardedLinear(nn.Module):
    """Linear layer with two algebraic output checksums and fault injection hooks."""
    def __init__(self, inp: int, out: int, name: str):
        super().__init__()
        self.linear = nn.Linear(inp, out)
        self.name = name
        self.inject = False
        self.faults = 1
        self.force_token_change = False
        self.last_detected = False
        self.last_verify_us = 0.0

    def forward(self, x, guard=True):
        clean = self.linear(x)
        observed = clean

        if self.inject:
            observed = clean.clone()
            flat = observed.reshape(-1, observed.shape[-1])
            row = flat.shape[0] - 1
            if self.force_token_change:
                winner = int(torch.argmax(flat[row]))
                choices = [i for i in range(flat.shape[1]) if i != winner]
                for k in range(self.faults):
                    col = choices[(7 + 11 * k) % len(choices)]
                    flat[row, col] = torch.max(flat[row]) + 50.0 + 10.0 * k
            else:
                for k in range(self.faults):
                    col = (13 + 17 * k) % flat.shape[1]
                    flat[row, col] += 1000.0 + 100.0 * k
            observed = flat.reshape_as(observed)

        t0 = time.perf_counter_ns()
        w, b = self.linear.weight, self.linear.bias
        nout = w.shape[0]
        c0 = torch.ones(nout, dtype=x.dtype, device=x.device)
        c1 = torch.arange(1, nout + 1, dtype=x.dtype, device=x.device)
        exp0 = x @ (w.T @ c0) + b @ c0
        exp1 = x @ (w.T @ c1) + b @ c1
        obs0 = observed @ c0
        obs1 = observed @ c1
        ok = torch.allclose(obs0, exp0, rtol=1e-4, atol=1e-4) and torch.allclose(obs1, exp1, rtol=1e-4, atol=1e-4)
        self.last_detected = not bool(ok)
        self.last_verify_us = (time.perf_counter_ns() - t0) / 1000.0

        if guard and self.last_detected:
            return self.linear(x)  # simulated peer recomputation
        return observed


class Block(nn.Module):
    def __init__(self, d=32, heads=4, ff=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = GuardedLinear(ff, d, "ffn_out")

    def forward(self, x, mask, guard=True):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        h = F.gelu(self.ff1(self.ln2(x)))
        return x + self.ff2(h, guard=guard)


class TinyLM(nn.Module):
    def __init__(self, vocab_size, max_len=16, d=32):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        self.block = Block(d=d)
        self.ln = nn.LayerNorm(d)
        self.head = GuardedLinear(d, vocab_size, "lm_head")

    def forward(self, ids, guard=True):
        _, t = ids.shape
        pos = torch.arange(t, device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None, :, :]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=ids.device), diagonal=1)
        x = self.block(x, mask, guard=guard)
        return self.head(self.ln(x), guard=guard)


SENTS = [
    "ORBIT COMPUTE RESULT IS SAFE .",
    "SPACE COMPUTE RESULT IS SAFE .",
    "NODE COMPUTE RESULT IS VALID .",
    "GPU COMPUTE RESULT IS VALID .",
]


def dataset():
    words = sorted(set(" ".join(SENTS).split()))
    vocab = ["<PAD>", "<BOS>", "<EOS>", "UNSAFE", "INVALID", "ERROR"] + [w for w in words if w not in {"UNSAFE", "INVALID", "ERROR"}]
    stoi = {w: i for i, w in enumerate(vocab)}
    seqs = []
    for _ in range(80):
        for s in SENTS:
            seqs.append([stoi[x] for x in ["<BOS>"] + s.split() + ["<EOS>"]])
    m = max(map(len, seqs)) - 1
    x, y = [], []
    for seq in seqs:
        x.append(seq[:-1] + [stoi["<PAD>"]] * (m - len(seq[:-1])))
        y.append(seq[1:] + [stoi["<PAD>"]] * (m - len(seq[1:])))
    return vocab, stoi, torch.tensor(x), torch.tensor(y)


def train(model, x, y, pad):
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    t0 = time.perf_counter()
    for step in range(500):
        model.block.ff2.inject = model.head.inject = False
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=pad)
        loss.backward(); opt.step()
        if float(loss) < 0.01:
            break
    return (time.perf_counter() - t0) * 1000, step + 1, float(loss)


@torch.no_grad()
def generate(model, vocab, stoi, prompt="ORBIT", guard=True, max_new=8):
    ids = [stoi[w] for w in prompt.split()]
    for _ in range(max_new):
        logits = model(torch.tensor([ids]), guard=guard)
        nxt = int(torch.argmax(logits[0, -1]))
        if vocab[nxt] == "<EOS>": break
        ids.append(nxt)
    return " ".join(vocab[i] for i in ids if vocab[i] not in {"<BOS>", "<EOS>", "<PAD>"})


@torch.no_grad()
def run_campaign(model, vocab, stoi):
    model.block.ff2.inject = model.head.inject = False
    clean = generate(model, vocab, stoi)
    clean_one = generate(model, vocab, stoi, max_new=1)

    fp = 0
    for _ in range(CLEAN_TRIALS):
        model.block.ff2.inject = model.head.inject = False
        generate(model, vocab, stoi)
        fp += int(model.head.last_detected or model.block.ff2.last_detected)

    det = rec = changed = 0
    times = []
    for _ in range(SINGLE_TRIALS):
        model.block.ff2.inject = False
        model.head.inject, model.head.faults, model.head.force_token_change = True, 1, True
        bad = generate(model, vocab, stoi, guard=False, max_new=1)
        changed += int(bad != clean_one)
        det += int(model.head.last_detected)
        good = generate(model, vocab, stoi, guard=True, max_new=1)
        rec += int(good == clean_one)
        times.append(model.head.last_verify_us)

    det2 = rec2 = 0
    for _ in range(DOUBLE_TRIALS):
        model.head.inject, model.head.faults, model.head.force_token_change = True, 2, True
        generate(model, vocab, stoi, guard=False, max_new=1)
        det2 += int(model.head.last_detected)
        rec2 += int(generate(model, vocab, stoi, guard=True, max_new=1) == clean_one)

    fdet = frec = 0
    for _ in range(FFN_TRIALS):
        model.head.inject = False
        model.block.ff2.inject, model.block.ff2.faults = True, 1
        generate(model, vocab, stoi, guard=False, max_new=1)
        fdet += int(model.block.ff2.last_detected)
        frec += int(generate(model, vocab, stoi, guard=True, max_new=1) == clean_one)

    model.block.ff2.inject = False
    model.head.inject, model.head.faults, model.head.force_token_change = True, 1, True
    without_guard = generate(model, vocab, stoi, guard=False, max_new=1)
    with_guard = generate(model, vocab, stoi, guard=True, max_new=1)
    model.head.inject = False

    return {
        "clean_text": clean, "without_guard_example": without_guard, "with_guard_example": with_guard,
        "clean_trials": CLEAN_TRIALS, "false_positives": fp,
        "logit_single_trials": SINGLE_TRIALS, "logit_single_changed_without_guard": changed,
        "logit_single_detected": det, "logit_single_recovered": rec,
        "logit_double_trials": DOUBLE_TRIALS, "logit_double_detected": det2, "logit_double_recovered": rec2,
        "ffn_single_trials": FFN_TRIALS, "ffn_single_detected": fdet, "ffn_single_recovered": frec,
        "mean_verify_us": sum(times) / len(times),
        "p95_verify_us": sorted(times)[int(0.95 * (len(times) - 1))],
        "status": "PASS" if fp == 0 and det == SINGLE_TRIALS and rec == SINGLE_TRIALS and det2 == DOUBLE_TRIALS and rec2 == DOUBLE_TRIALS and fdet == FFN_TRIALS and frec == FFN_TRIALS else "CHECK",
    }


def report(r, train_ms, steps, loss):
    def pct(a, b): return f"{100*a/b:.2f}%"
    body = f"""<!doctype html><meta charset='utf-8'><title>STARXION v1</title><style>
body{{font-family:Arial;background:#0b0f14;color:#e8edf2;margin:0;padding:36px}}.wrap{{max-width:1000px;margin:auto}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#121923;border:1px solid #263241;border-radius:14px;padding:20px}}
.big{{font-size:32px;font-weight:800}}small{{color:#9fb0c0}}.demo{{padding:18px;border:1px solid #263241;border-radius:12px;margin:20px 0;font-family:Consolas}}
</style><div class='wrap'><h1>STARXION v1 — {r['status']}</h1><p>Silent-compute integrity for AI systems</p>
<div class='demo'>WITHOUT GUARD: {html.escape(r['without_guard_example'])}<br>WITH STARXION: {html.escape(r['with_guard_example'])}</div>
<div class='grid'>
<div class='card'><small>Single fault detection</small><div class='big'>{pct(r['logit_single_detected'],r['logit_single_trials'])}</div></div>
<div class='card'><small>Double fault detection</small><div class='big'>{pct(r['logit_double_detected'],r['logit_double_trials'])}</div></div>
<div class='card'><small>Internal FFN detection</small><div class='big'>{pct(r['ffn_single_detected'],r['ffn_single_trials'])}</div></div>
<div class='card'><small>Recovery</small><div class='big'>{pct(r['logit_single_recovered'],r['logit_single_trials'])}</div></div>
<div class='card'><small>False positives</small><div class='big'>{r['false_positives']}</div></div>
<div class='card'><small>Mean verification</small><div class='big'>{r['mean_verify_us']:.1f} μs</div></div></div>
<p><b>Reference environment:</b> PyTorch {torch.__version__}; training {steps} steps / {train_ms:.1f} ms; loss {loss:.5f}</p>
<p><b>Scope:</b> software-injected faults; local simulated peer recomputation; not radiation-tested or flight-qualified.</p></div>"""
    Path("starxion_report.html").write_text(body, encoding="utf-8")


def main():
    vocab, stoi, x, y = dataset()
    model = TinyLM(len(vocab))
    train_ms, steps, loss = train(model, x, y, stoi["<PAD>"])
    model.eval()
    r = run_campaign(model, vocab, stoi)
    r.update(torch_version=torch.__version__, train_ms=train_ms, train_steps=steps, train_loss=loss)
    Path("starxion_results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    report(r, train_ms, steps, loss)
    print("\n=== STARXION v1 ===")
    print("Status:", r["status"])
    print("WITHOUT GUARD:", r["without_guard_example"])
    print("WITH STARXION:", r["with_guard_example"])
    print(f"Single faults: {r['logit_single_detected']}/{r['logit_single_trials']} detected")
    print(f"Double faults: {r['logit_double_detected']}/{r['logit_double_trials']} detected")
    print(f"Internal FFN faults: {r['ffn_single_detected']}/{r['ffn_single_trials']} detected")
    print(f"False positives: {r['false_positives']}/{r['clean_trials']}")

if __name__ == "__main__":
    main()

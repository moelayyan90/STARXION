# STARXION

**Trust every computation beyond Earth.**

STARXION is an experimental AI-compute integrity layer for detecting silent numerical corruption during neural-network inference before a corrupted result is accepted.

The current proof of concept wraps Transformer-style linear operations with lightweight algebraic checks, injects synthetic silent faults, rejects corrupted outputs, and recomputes the affected result.

## Current benchmark

- PyTorch Transformer-style block: MultiheadAttention + LayerNorm + GELU FFN
- 1,000 / 1,000 injected single-logit faults detected
- 500 / 500 injected double-logit faults detected
- 500 / 500 injected internal FFN faults detected
- 1,000 / 1,000 corrupted output cases recovered after rejection
- 0 false positives across 300 clean checks

The raw result from the reference run is included in [`starxion_results.json`](starxion_results.json).

## Run it

### Windows

Double-click:

```text
RUN_STARXION.bat
```

### Command line

```bash
python -m pip install -r requirements.txt
python starxion.py
```

The benchmark writes:

- `starxion_results.json`
- `starxion_report.html`

## How it works

For a protected linear operation, STARXION computes two independent algebraic checksum relations alongside the normal inference path. A mismatch marks the output as corrupted before acceptance. The current demo then simulates peer recovery by recomputing the affected operation locally.

## Why this exists

AI inference deployed in fault-prone compute environments needs a way to distinguish a plausible-looking model result from a numerically corrupted computation. STARXION explores a software layer for that problem with low-cost verification and selective recomputation.

## Scope and limitations

This repository is an **engineering proof of concept**.

- Faults are **software-injected**, not radiation-induced.
- Peer recomputation is currently **simulated on the same machine**.
- The project has **not** been tested under a radiation beam or on spacecraft hardware.
- It is **not flight-qualified** and **not a production LLM runtime**.
- The reported 100% rates apply only to the injected fault campaign in this repository; they are not a general reliability guarantee.

## Next validation targets

1. Pretrained Transformer weights rather than the tiny locally trained model.
2. Real multi-device peer recomputation.
3. GPU-level fault injection and larger fault distributions.
4. Radiation/accelerator testing with hardware telemetry.
5. Measurement of end-to-end latency, energy and bandwidth overhead.

---

**STARXION — silent-compute integrity for AI systems.**

# STARXION

## Trust every computation beyond Earth.

### One-line thesis

STARXION is a software integrity layer that detects silent numerical corruption during AI inference before a corrupted result is accepted, then selectively recomputes only the affected work.

### Why it matters

Spaceborne AI compute must tolerate fault-prone environments without assuming that every numerical error will crash visibly. A silent error can instead propagate into a plausible-looking model output.

### What exists now

A reproducible PyTorch proof of concept with automated CI. The current synthetic fault campaign detects:

- 1,000 / 1,000 single injected logit faults;
- 500 / 500 double injected logit faults;
- 500 / 500 internal FFN faults;
- 0 false positives across 300 clean checks.

When corruption is detected, the affected operation is rejected and recomputed.

### What is novel in the product direction

The intended architecture is not full duplicate inference. STARXION aims to combine low-cost verification with selective peer recomputation so that redundancy is paid for only when the integrity check fails.

### Next validation milestone

Run the same mechanism on pretrained Transformer inference across two independent devices, inject GPU-level faults, and measure detection coverage plus end-to-end latency, energy and bandwidth overhead.

### Current status

Engineering proof of concept only. Software-injected faults; no radiation testing; no flight qualification.

**Repository:** `moelayyan90/STARXION`

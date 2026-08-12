# STARXION Technical Note

## Problem

Neural-network inference can produce a numerically corrupted result that still looks structurally valid to downstream software. In fault-prone compute environments, that creates a failure mode where a corrupted tensor can be accepted as if it were legitimate.

## Approach

STARXION wraps selected linear operations with lightweight algebraic verification. For a protected operation, two independent checksum relations are evaluated alongside the normal inference path. If either relation fails, the result is rejected before acceptance and the affected computation is recomputed.

The present repository demonstrates three classes of software-injected faults:

- single corrupted output-logit values;
- double corrupted output-logit values;
- corruption inside an internal feed-forward-network output.

## Reference benchmark

The current reference run reports:

- 1,000 / 1,000 single-logit faults detected;
- 500 / 500 double-logit faults detected;
- 500 / 500 internal FFN faults detected;
- 1,000 / 1,000 corrupted output cases recovered after rejection;
- 0 false positives across 300 clean checks.

These numbers describe only the synthetic fault campaign in this repository. They are not a general reliability guarantee.

## Recovery model

The current implementation simulates an independent peer by recomputing the affected operation locally after a checksum failure. A constellation-oriented implementation would instead route the rejected operation to an independent device or node, then compare the recomputed result before acceptance.

## What still has to be proven

Before STARXION could be considered for real spaceborne AI workloads, it would need at minimum:

1. evaluation on pretrained production-scale Transformer inference;
2. GPU-level fault injection;
3. real multi-device peer recomputation;
4. measurements of latency, bandwidth, energy and memory overhead;
5. hardware radiation testing with telemetry;
6. adversarial and correlated-fault analysis;
7. flight-software integration and qualification.

## Design objective

The target is not to duplicate every inference operation. The target is to make silent corruption observable cheaply enough that expensive recomputation is triggered only when verification fails.

**STARXION — Trust every computation beyond Earth.**
